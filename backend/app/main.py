from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, teams

app = FastAPI(title="NBA Playoff Experience API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(teams.router, prefix="/teams", tags=["teams"])
