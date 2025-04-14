import os
import json
import datetime
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Get database URL from environment variable or use SQLite as fallback
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("DATABASE_URL environment variable is not set. Using SQLite as fallback.")
    DATABASE_URL = "sqlite:///./memotag.db"

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define database models
class User(Base):
    """User model for authentication and tracking user's analyses."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship to analyses
    analyses = relationship("Analysis", back_populates="user")

class Analysis(Base):
    """Analysis model for storing voice analysis results."""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Audio metadata
    audio_duration = Column(Float)
    audio_sample_rate = Column(Integer)
    audio_format = Column(String)
    
    # Transcription results
    transcription_text = Column(Text)
    transcription_json = Column(JSON, nullable=True)
    
    # Analysis results
    features_json = Column(JSON, nullable=True)
    analysis_results_json = Column(JSON, nullable=True)
    risk_score = Column(Float, nullable=True)
    
    # Report
    report_html = Column(Text, nullable=True)
    
    # Relationship to user
    user = relationship("User", back_populates="analyses")
    
    # Relationship to markers
    cognitive_markers = relationship("CognitiveMarker", back_populates="analysis")

class CognitiveMarker(Base):
    """Cognitive marker model for storing identified cognitive markers in an analysis."""
    __tablename__ = "cognitive_markers"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    marker_name = Column(String)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float)
    
    # Relationship to analysis
    analysis = relationship("Analysis", back_populates="cognitive_markers")

# Create all tables
Base.metadata.create_all(bind=engine)

# Database helper functions
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_user(username: str, email: str) -> User:
    """Create a new user."""
    db = SessionLocal()
    try:
        user = User(username=username, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()

def get_user_by_username(username: str) -> Optional[User]:
    """Get a user by username."""
    db = SessionLocal()
    try:
        return db.query(User).filter(User.username == username).first()
    finally:
        db.close()

def get_user_by_id(user_id: int) -> Optional[User]:
    """Get a user by ID."""
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()

def save_analysis(
    user_id: Optional[int],
    title: str,
    audio_info: Dict[str, Any],
    transcription: Dict[str, Any],
    features: Dict[str, Any],
    analysis_results: Dict[str, Any],
    report_html: str,
    description: Optional[str] = None
) -> Analysis:
    """Save analysis results to the database."""
    db = SessionLocal()
    try:
        # Extract risk score from analysis results
        risk_score = None
        if 'anomalies' in analysis_results and 'risk_score' in analysis_results['anomalies']:
            risk_score = analysis_results['anomalies']['risk_score']
        
        # Create analysis record
        analysis = Analysis(
            user_id=user_id,
            title=title,
            description=description,
            audio_duration=audio_info.get('duration', 0),
            audio_sample_rate=audio_info.get('sample_rate', 0),
            audio_format=audio_info.get('format', ''),
            transcription_text=transcription.get('text', ''),
            transcription_json=transcription,
            features_json=features,
            analysis_results_json=analysis_results,
            risk_score=risk_score,
            report_html=report_html
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        # Save cognitive markers if available
        if 'anomalies' in analysis_results and 'markers' in analysis_results['anomalies']:
            for marker_name, marker_data in analysis_results['anomalies']['markers'].items():
                marker = CognitiveMarker(
                    analysis_id=analysis.id,
                    marker_name=marker_name,
                    is_anomaly=marker_data.get('is_outlier', False),
                    anomaly_score=marker_data.get('anomaly_score', 0)
                )
                db.add(marker)
            
            db.commit()
        
        return analysis
    finally:
        db.close()

def get_analysis_by_id(analysis_id: int) -> Optional[Analysis]:
    """Get an analysis by ID."""
    db = SessionLocal()
    try:
        return db.query(Analysis).filter(Analysis.id == analysis_id).first()
    finally:
        db.close()

def get_analyses_by_user(user_id: int) -> List[Analysis]:
    """Get all analyses for a user."""
    db = SessionLocal()
    try:
        return db.query(Analysis).filter(Analysis.user_id == user_id).all()
    finally:
        db.close()

def get_all_analyses() -> List[Analysis]:
    """Get all analyses."""
    db = SessionLocal()
    try:
        return db.query(Analysis).all()
    finally:
        db.close()