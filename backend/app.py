"""
OptiFleet Server Module (backend/app.py)
----------------------------------------
FastAPI Web Framework Server providing:
1. REST API Endpoint (/api/optimize) for trigger-based matching & pricing
2. Real-Time WebSocket Endpoint (/ws/simulation) for continuous state updates
3. Pydantic Request/Response Schema Validation
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json

from engine import FleetOptimizationEngine, Driver, CustomerRequest, Location

app = FastAPI(
    title="OptiFleet Optimization API Server",
    description="Backend Service powering IIT Patna Optimization Term Project",
    version="1.0.0"
)

# Enable CORS for local web browsers and frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global optimization engine instance
engine = FleetOptimizationEngine()

# --- PYDANTIC SCHEMAS ---
class LocationSchema(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

class DriverSchema(BaseModel):
    id: int
    location: LocationSchema
    available: bool = True

class RequestSchema(BaseModel):
    id: int
    location: LocationSchema

class OptimizationPayload(BaseModel):
    drivers: List[DriverSchema]
    requests: List[RequestSchema]
    surge_aggression: float = Field(1.5, ge=1.0, le=2.5)
    max_detour_km: float = Field(3.5, ge=1.0, le=10.0)

class OptimizationResponse(BaseModel):
    status: str
    active_matches_count: int
    matches: List[Dict[str, Any]]
    pricing_info: Dict[str, float]
    average_wait_time_min: float

# --- REST ENDPOINTS ---
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "OptiFleet Optimization Engine",
        "campus": "IIT Patna (Bihta)"
    }

@app.post("/api/optimize", response_model=OptimizationResponse)
def run_fleet_optimization(payload: OptimizationPayload):
    """
    Main trigger endpoint: Takes current driver and rider positions, computes
    optimal bipartite matching and computes Stackelberg dynamic surge multipliers.
    """
    # 1. Convert incoming JSON schemas into internal Engine dataclasses
    engine_drivers = [
        Driver(id=d.id, location=Location(x=d.location.x, y=d.location.y), available=d.available)
        for d in payload.drivers
    ]
    engine_requests = [
        CustomerRequest(id=r.id, location=Location(x=r.location.x, y=r.location.y))
        for r in payload.requests
    ]

    # 2. Run Maximum-Weight Bipartite Matcher
    matches = engine.solve_bipartite_matching(
        drivers=engine_drivers,
        requests=engine_requests,
        max_detour_km=payload.max_detour_km
    )

    # 3. Calculate Stackelberg Equilibrium Dynamic Price Surge
    pricing = engine.calculate_stackelberg_surge(
        active_requests_count=len(engine_requests),
        available_drivers_count=len(engine_drivers),
        aggressiveness=payload.surge_aggression
    )

    # 4. Compute Aggregate Metrics
    avg_wait = (
        sum(m["eta_minutes"] for m in matches) / len(matches) 
        if matches else 0.0
    )

    return {
        "status": "success",
        "active_matches_count": len(matches),
        "matches": matches,
        "pricing_info": pricing,
        "average_wait_time_min": round(avg_wait, 1)
    }

# --- WEBSOCKET LIVE STREAMING ENDPOINT ---
@app.websocket("/ws/simulation")
async def websocket_simulation_endpoint(websocket: WebSocket):
    """
    WebSocket route broadcasting live positions of drivers and dynamic matching updates every 1 second.
    """
    await websocket.accept()
    
    # Initialize mock fleet state
    drivers = [
        Driver(id=101+i, location=Location(x=0.2 + (i*0.05), y=0.3 + (i*0.03)))
        for i in range(15)
    ]
    
    try:
        while True:
            # Step simulation forward in time
            drivers = engine.update_simulation_step(drivers)
            
            # Broadcast state packet
            packet = {
                "type": "FLEET_UPDATE",
                "drivers": [
                    {"id": d.id, "x": d.location.x, "y": d.location.y}
                    for d in drivers
                ]
            }
            await websocket.send_text(json.dumps(packet))
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)