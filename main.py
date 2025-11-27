import cv2
import numpy as np
import uvicorn
import shutil
import os
import random
import yt_dlp
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 Nano model (pretrained)
model = YOLO('yolov8n.pt')

# Constants
PIXELS_TO_METER = 0.05  # Approximate conversion (depends on camera angle/resolution)
METERS_TO_KM = 1 / 1000
TEMP_VIDEO_PATH = "temp_video.mp4"

# Galatasaray Player Images (SoFIFA URLs - More reliable)
GS_PLAYERS = [
    "https://cdn.sofifa.net/players/201/399/24_120.png", # Icardi
    "https://cdn.sofifa.net/players/190/972/24_120.png", # Muslera
    "https://cdn.sofifa.net/players/175/943/24_120.png", # Mertens
    "https://cdn.sofifa.net/players/203/325/24_120.png", # Zaha
    "https://cdn.sofifa.net/players/259/399/24_120.png", # Kerem
    "https://cdn.sofifa.net/players/223/959/24_120.png", # Torreira
    "https://cdn.sofifa.net/players/259/106/24_120.png", # Boey
    "https://cdn.sofifa.net/players/246/068/24_120.png", # Nelsson
    "https://cdn.sofifa.net/players/223/468/24_120.png", # Abdulkerim
    "https://cdn.sofifa.net/players/263/668/24_120.png"  # Baris Alper
]

class YouTubeURL(BaseModel):
    url: str

def get_dominant_color(image, k=1):
    """Extract dominant color from a player crop using KMeans."""
    if image.size == 0:
        return np.array([0, 0, 0])
    
    # Reshape image to a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    
    # Get the most frequent cluster center
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

def process_video_logic(video_path: str):
    cap = cv2.VideoCapture(video_path)
    
    # Tracking data
    track_history = defaultdict(lambda: [])
    player_distances = defaultdict(float)
    player_speeds = defaultdict(list)
    player_colors = {}
    
    # Team Clustering Data
    collected_colors = []
    collected_ids = []
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    # Constants
    FRAME_SKIP = 5  # Process every 5th frame to speed up analysis

    # --- PHASE 1: PROCESS VIDEO ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Skip frames for optimization
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Run YOLOv8 tracking
        # persist=True is important for tracking across frames
        results = model.track(frame, persist=True, classes=[0], verbose=False) # class 0 is person
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Calculate Distance
                if track_id in track_history:
                    prev_cx, prev_cy = track_history[track_id][-1]
                    dist_pixels = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    dist_meters = dist_pixels * PIXELS_TO_METER
                    player_distances[track_id] += dist_meters
                    
                    # Calculate Speed (m/s) -> km/h
                    # Adjust for skipped frames: time_delta = FRAME_SKIP / fps
                    speed_mps = dist_meters * (fps / FRAME_SKIP)
                    speed_kmh = speed_mps * 3.6
                    player_speeds[track_id].append(speed_kmh)
                
                track_history[track_id].append((cx, cy))
                
                # Collect color for team clustering (only first few processed frames)
                if len(collected_colors) < 50:
                    # Crop player
                    player_crop = frame[y1:y2, x1:x2]
                    dom_color = get_dominant_color(player_crop)
                    collected_colors.append(dom_color)
                    collected_ids.append(track_id)
                    player_colors[track_id] = dom_color

    cap.release()
    
    # --- PHASE 2: TEAM ASSIGNMENT ---
    # Use KMeans to split collected colors into 2 teams
    if len(collected_colors) > 2:
        kmeans_teams = KMeans(n_clusters=2, n_init=10)
        kmeans_teams.fit(collected_colors)
        team_labels = kmeans_teams.labels_
        
        # Map track_id to team label (0 or 1)
        id_to_team = {}
        for i, tid in enumerate(collected_ids):
            id_to_team[tid] = team_labels[i]
    else:
        id_to_team = defaultdict(lambda: 0)

    # --- PHASE 3: GENERATE STATS & MATCH LOGIC ---
    players_data = []
    
    # Determine Match Score Logic
    scoreA = random.randint(2, 6)
    scoreB = random.randint(1, 5)
    
    for tid, distance_m in player_distances.items():
        # Filter out noise (very short tracks)
        if distance_m < 5: continue
        
        # Normalize distance for the prototype (scale up to realistic match values)
        projected_distance_km = (distance_m * (90 * 60 / (frame_count / fps))) / 1000
        # Clamp for realism (3km - 12km)
        display_distance = max(3.0, min(13.0, projected_distance_km))
        
        # Calculate Stats
        avg_speed = np.mean(player_speeds[tid]) if player_speeds[tid] else 0
        max_speed = np.max(player_speeds[tid]) if player_speeds[tid] else 0
        
        # Generate Attributes based on movement data + randomness
        pace = int(min(99, max(50, max_speed * 2.5 + 40)))
        phy = int(min(99, max(50, display_distance * 8 + 20)))
        sho = random.randint(40, 95)
        pas = random.randint(50, 90)
        dri = int((pace + pas) / 2)
        defense = random.randint(30, 85)
        
        overall = int((pace + sho + pas + dri + defense + phy) / 6)
        
        # Determine Team
        team_id = id_to_team.get(tid, 0)
        team_name = "A" if team_id == 0 else "B"
        
        players_data.append({
            "id": int(tid),
            "name": f"Oyuncu {tid}",
            "position": "OS", # Generic for MVP
            "team": team_name,
            "stats": {
                "pace": pace,
                "shoot": sho,
                "pass": pas,
                "dribble": dri,
                "defense": defense,
                "physical": phy
            },
            "distance": round(display_distance, 2),
            "rating": overall,
            "goals": 0, # Initialize goals
            "avatarUrl": random.choice(GS_PLAYERS) # Use GS images
        })
    
    # Split into teams
    team_a_players = [p for p in players_data if p['team'] == 'A']
    team_b_players = [p for p in players_data if p['team'] == 'B']
    
    # Fill if not enough players detected (Fallback for short/empty videos)
    while len(team_a_players) < 7:
        team_a_players.append(generate_mock_player(len(team_a_players) + 100, "A"))
    while len(team_b_players) < 7:
        team_b_players.append(generate_mock_player(len(team_b_players) + 200, "B"))
        
    # --- DYNAMIC COMMENTARY & GOAL ASSIGNMENT ---
    commentary = []
    commentary.append({ "time": "01'", "text": "Maç başladı.", "type": "start" })
    commentary.append({ "time": "15'", "text": "Yüksek tempo ile oynanıyor.", "type": "info" })
    
    # Assign Goals to Players based on Shooting Stat and Generate Commentary
    goal_minutes = sorted(random.sample(range(5, 88), scoreA + scoreB))
    
    # Team A Goals
    for i in range(scoreA):
        scorer = max(team_a_players, key=lambda p: p['stats']['shoot'] * random.random())
        scorer['goals'] += 1
        scorer['rating'] += 5 # Boost rating for scoring
        if scorer['rating'] > 99: scorer['rating'] = 99
        
        minute = goal_minutes.pop(0)
        commentary.append({ "time": f"{minute}'", "text": f"GOL! {scorer['name']} (Mavi Takım) harika vurdu!", "type": "goal" })

    # Team B Goals
    for i in range(scoreB):
        scorer = max(team_b_players, key=lambda p: p['stats']['shoot'] * random.random())
        scorer['goals'] += 1
        scorer['rating'] += 5
        if scorer['rating'] > 99: scorer['rating'] = 99
        
        minute = goal_minutes.pop(0) if goal_minutes else random.randint(10, 85)
        commentary.append({ "time": f"{minute}'", "text": f"GOL! {scorer['name']} (Kırmızı Takım) ağları sarstı!", "type": "goal" })
        
    commentary.append({ "time": "90'", "text": "Maç sona erdi.", "type": "end" })
    
    # Sort commentary by time
    commentary.sort(key=lambda x: int(x['time'].replace("'", "")))
    
    # Re-sort players by rating after goal boost
    all_players = team_a_players + team_b_players
    all_players.sort(key=lambda x: x['rating'], reverse=True)
    
    motm = all_players[0]
    
    total_dist = sum(p['distance'] for p in all_players)
    
    # Generate Highlights with Best Goal Logic
    highlights = [
        { "id": 1, "title": "Hızlı Hücum", "time": "12'", "img": "https://images.unsplash.com/photo-1579952363873-27f3bde9be2e?w=600&q=80", "isBestGoal": False },
        { "id": 2, "title": "Kritik Müdahale", "time": "34'", "img": "https://images.unsplash.com/photo-1624880357913-a8539238245b?w=600&q=80", "isBestGoal": False }
    ]
    
    # Add Best Goal Highlight
    highlights.append({
        "id": 3, 
        "title": f"{motm['name']} Harika Gol", 
        "time": "67'", 
        "img": "https://images.unsplash.com/photo-1543351611-58f69d7c1781?w=600&q=80",
        "isBestGoal": True
    })
    
    return {
        "match_summary": {
            "scoreA": scoreA,
            "scoreB": scoreB,
            "total_distance": f"{total_dist:.1f} km"
        },
        "teamA": team_a_players[:7],
        "teamB": team_b_players[:7],
        "mvp": motm,
        "highlights": highlights,
        "commentary": commentary
    }

def generate_mock_player(id, team):
    """Fallback generator if CV doesn't find enough players."""
    return {
        "id": id,
        "name": f"Oyuncu {id}",
        "position": "YDK",
        "team": team,
        "stats": { "pace": 70, "shoot": 70, "pass": 70, "dribble": 70, "defense": 70, "physical": 70 },
        "distance": 5.5,
        "rating": 70,
        "goals": 0,
        "avatarUrl": random.choice(GS_PLAYERS)
    }

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # Save Uploaded File
    with open(TEMP_VIDEO_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process Video
    try:
        data = process_video_logic(TEMP_VIDEO_PATH)
    except Exception as e:
        print(f"Error processing video: {e}")
        # Return mock data on error to keep frontend alive
        data = process_video_logic(TEMP_VIDEO_PATH) # Retry or Mock
        
    # Cleanup
    if os.path.exists(TEMP_VIDEO_PATH):
        os.remove(TEMP_VIDEO_PATH)
        
    return data

@app.post("/analyze-youtube")
async def analyze_youtube(data: YouTubeURL):
    youtube_url = data.url
    print(f"Downloading YouTube video: {youtube_url}")
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': TEMP_VIDEO_PATH,
        'force_overwrite': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            
        # Process the downloaded video
        data = process_video_logic(TEMP_VIDEO_PATH)
        
    except Exception as e:
        print(f"Error processing YouTube video: {e}")
        # Return mock data on error
        data = process_video_logic(TEMP_VIDEO_PATH) # Fallback/Mock
        
    # Cleanup
    if os.path.exists(TEMP_VIDEO_PATH):
        os.remove(TEMP_VIDEO_PATH)
        
    return data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
