"""
OptiFleet Engine Module (backend/engine.py)
--------------------------------------------
Implements:
1. Distance Matrix Computation (Geographic Euclidean / Manhattan metric)
2. Maximum Weight Bipartite Matching via Scipy Hungarian Method
3. Stackelberg Leader-Follower Dynamic Surge Pricing Optimization
4. Real-time Spatial Fleet Simulation Loop
"""

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any

@dataclass
class Location:
    x: float  # Normalized 0.0 - 1.0 or Lat/Lng coordinate
    y: float

@dataclass
class Driver:
    id: int
    location: Location
    available: bool = True
    speed: float = 0.002

@dataclass
class CustomerRequest:
    id: int
    location: Location
    time_window_min: float = 5.0

class FleetOptimizationEngine:
    def __init__(self, base_fare: float = 40.0, per_km_rate: float = 12.0):
        self.base_fare = base_fare
        self.per_km_rate = per_km_rate

    def compute_cost_matrix(self, drivers: List[Driver], requests: List[CustomerRequest]) -> np.ndarray:
        """
        Builds the n x m Cost Matrix C where C[i, j] represents the estimated wait time / distance
        between Driver i and Customer Request j.
        """
        n_drivers = len(drivers)
        n_requests = len(requests)
        cost_matrix = np.zeros((n_drivers, n_requests))

        for i, driver in enumerate(drivers):
            for j, req in enumerate(requests):
                # Euclidean distance calculation scaled to campus grid km (1 unit ≈ 5 km)
                dx = driver.location.x - req.location.x
                dy = driver.location.y - req.location.y
                dist_km = np.sqrt(dx**2 + dy**2) * 5.0
                
                # Cost function balances distance penalty with vehicle availability
                cost_matrix[i, j] = dist_km
                
        return cost_matrix

    def solve_bipartite_matching(
        self, 
        drivers: List[Driver], 
        requests: List[CustomerRequest], 
        max_detour_km: float = 3.5
    ) -> List[Dict[str, Any]]:
        """
        Solves Min-Cost Maximum-Cardinality Bipartite Matching using the Hungarian Algorithm
        O(N^3) time complexity using SciPy's linear_sum_assignment.
        """
        if not drivers or not requests:
            return []

        cost_matrix = self.compute_cost_matrix(drivers, requests)
        driver_indices, request_indices = linear_sum_assignment(cost_matrix)

        matches = []
        for d_idx, r_idx in zip(driver_indices, request_indices):
            dist_km = cost_matrix[d_idx, r_idx]
            
            # Filter matches exceeding max allowable detour constraint
            if dist_km <= max_detour_km:
                eta_minutes = round((dist_km / 20.0) * 60, 1) + 1.5  # Assumes 20km/h avg speed + 1.5m prep
                matches.append({
                    "driver_id": drivers[d_idx].id,
                    "request_id": requests[r_idx].id,
                    "distance_km": round(float(dist_km), 2),
                    "eta_minutes": eta_minutes,
                    "driver_loc": asdict(drivers[d_idx].location),
                    "request_loc": asdict(requests[r_idx].location)
                })

        return matches

    def calculate_stackelberg_surge(
        self, 
        active_requests_count: int, 
        available_drivers_count: int, 
        aggressiveness: float = 1.5
    ) -> Dict[str, float]:
        """
        Models Platform Dynamic Surge Pricing as a Stackelberg Game:
        - Platform (Leader) sets Surge Multiplier (S)
        - Riders & Drivers (Followers) react according to price elasticity functions.
        """
        if available_drivers_count == 0:
            return {"surge_multiplier": 2.5, "expected_revenue": 0.0, "demand_fulfilled_ratio": 0.0}

        # Ratio of supply to demand
        supply_demand_ratio = available_drivers_count / max(1, active_requests_count)

        # Objective Function: Maximize Platform Profit (Negative for SciPy minimization)
        def negative_profit_objective(surge_array):
            s = surge_array[0]
            
            # Demand elasticity curve (Logistic decay with higher surge)
            effective_demand = active_requests_count / (1.0 + np.exp(0.9 * (s - 1.2 * aggressiveness)))
            
            # Supply elasticity curve (More drivers activate at higher rates)
            effective_supply = available_drivers_count * (1.0 + 0.3 * (s - 1.0))
            
            # Realized completed trips
            trips = min(effective_demand, effective_supply)
            fare_per_trip = (self.base_fare + 2.5 * self.per_km_rate) * s
            platform_revenue = trips * fare_per_trip * 0.20  # 20% platform commission rate
            
            return -platform_revenue

        # Optimization bounded between 1.0x and 3.0x surge
        result = minimize(negative_profit_objective, x0=[1.2], bounds=[(1.0, 3.0)], method='L-BFGS-B')
        optimal_surge = round(float(result.x[0]), 2)
        expected_revenue = round(float(-result.fun), 2)

        return {
            "surge_multiplier": optimal_surge,
            "expected_revenue": expected_revenue,
            "supply_demand_ratio": round(supply_demand_ratio, 2)
        }

    def update_simulation_step(self, drivers: List[Driver]) -> List[Driver]:
        """
        Updates kinematic positions of unassigned vehicles across the campus grid.
        """
        for d in drivers:
            d.location.x += d.vx if hasattr(d, 'vx') else (np.random.rand() - 0.5) * d.speed
            d.location.y += d.vy if hasattr(d, 'vy') else (np.random.rand() - 0.5) * d.speed
            
            # Bounce off map bounding limits
            d.location.x = float(np.clip(d.location.x, 0.05, 0.95))
            d.location.y = float(np.clip(d.location.y, 0.05, 0.95))
            
        return drivers