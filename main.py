"""
Entry point for the API backend. 
Handles incoming HTTP requests, validates the parameters, and triggers the optimization.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from optimizer import run_optimization

app = FastAPI(title="Stiffened Plate Optimizer API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class OptimizationRequest(BaseModel):
    plate_length:  float = 1524.0
    plate_breadth: float = 762.0
    tp:            float = 20.0
    q_load:        float = 1.0
    BC:            int   = 1    # 1 = Fixed/Clamped, 0 = Simply Supported


@app.post("/optimize")
def optimize_plate(payload: OptimizationRequest):
    """
    Takes the requested plate dimensions and loads from the frontend, 
    checks if they are valid positive numbers, and runs the optimizer.
    """
    if payload.plate_length <= 0:
        raise HTTPException(status_code=400, detail="Plate length must be positive.")
    if payload.plate_breadth <= 0:
        raise HTTPException(status_code=400, detail="Plate breadth must be positive.")
    if payload.tp <= 0:
        raise HTTPException(status_code=400, detail="Initial plate thickness must be positive.")
    if payload.q_load < 0:
        raise HTTPException(status_code=400, detail="Uniform load cannot be negative.")
    if payload.BC not in (0, 1):
        raise HTTPException(status_code=400, detail="Boundary condition must be 0 or 1.")

    try:
        result = run_optimization(
            bc=payload.BC,
            plate_length=payload.plate_length,
            plate_breadth=payload.plate_breadth,
            initial_thickness=payload.tp,
            q_load=payload.q_load,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)