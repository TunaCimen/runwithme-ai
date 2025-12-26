"""
Inference service for computing route embeddings on-demand.
This is used for new routes that aren't in the pre-computed index.

Usage:
    1. As HTTP service: python inference_service.py --serve --port 8000
    2. As CLI: python inference_service.py --route route_data.json
    3. As Python module: from inference_service import RouteInferenceService
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
import argparse
import sys

# Import the model and feature registry
from modular_features import (
    HybridRouteModelModular, 
    FeatureRegistry,
    RouteCNN,
    MetadataEncoderModular,
    TERRAIN_TYPES,
    RUNNER_TYPES,
    DAYS_OF_WEEK,
    TIME_OF_DAY_CATEGORIES
)


class RouteInferenceService:
    """Service for computing route embeddings on-demand."""
    
    def __init__(self, export_dir: str = "export_kotlin"):
        self.export_dir = Path(export_dir)
        self.model = None
        self.feature_names = None
        self.embedding_dim = 128
        self.device = torch.device('cpu')  # Use CPU for low-resource deployment
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from export directory."""
        model_path = self.export_dir / "model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.feature_names = checkpoint.get('feature_names', list(FeatureRegistry.get_all_features().keys()))
        self.embedding_dim = checkpoint.get('embedding_dim', 128)
        
        # Import the model components from modular_features
        
        # Create sub-models
        cnn_model = RouteCNN(embedding_dim=128)
        metadata_model = MetadataEncoderModular(fusion_dim=128, embedding_dim=32)
        
        # Create full model
        self.model = HybridRouteModelModular(cnn_model, metadata_model, fusion_dim=128)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Features: {self.feature_names}")
        print(f"   Embedding dim: {self.embedding_dim}")
    
    def _prepare_route_tensor(self, route_data: Dict) -> Dict[str, torch.Tensor]:
        """Prepare route data as tensors for model input."""
        # Extract features using registry
        features_raw = FeatureRegistry.extract_features(route_data)
        
        # Convert to tensors using registry's tensor converters
        feature_tensors = FeatureRegistry.convert_to_tensors(features_raw)
        
        # Add batch dimension to each tensor
        for name in feature_tensors:
            feature_tensors[name] = feature_tensors[name].unsqueeze(0).to(self.device)
        
        # Prepare path image tensor
        path_image = self._create_path_image(route_data)
        feature_tensors['path_image'] = path_image.unsqueeze(0).to(self.device)
        
        return feature_tensors
    
    def _create_path_image(self, route_data: Dict, image_size: int = 64) -> torch.Tensor:
        """Create a path image from route geometry."""
        # Extract points from route data
        geometry = route_data.get('Geometry', {})
        
        if isinstance(geometry, str):
            try:
                geometry = json.loads(geometry)
            except:
                geometry = {}
        
        coordinates = geometry.get('coordinates', [])
        
        if not coordinates or len(coordinates) < 2:
            # Return blank image if no geometry
            return torch.zeros(1, image_size, image_size)
        
        # Convert to numpy array
        points = np.array(coordinates)
        
        # Normalize to [0, 1] range
        if len(points) > 0:
            min_vals = points.min(axis=0)
            max_vals = points.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            
            normalized = (points - min_vals) / range_vals
            
            # Scale to image coordinates with padding
            padding = 2
            scale = image_size - 2 * padding
            img_coords = (normalized * scale + padding).astype(int)
            img_coords = np.clip(img_coords, 0, image_size - 1)
        else:
            img_coords = np.array([[image_size // 2, image_size // 2]])
        
        # Create image
        image = np.zeros((image_size, image_size), dtype=np.float32)
        
        # Draw path
        for i in range(len(img_coords) - 1):
            x0, y0 = img_coords[i][:2]  # Take only first 2 coords (x, y)
            x1, y1 = img_coords[i + 1][:2]
            
            # Simple line drawing (Bresenham-like)
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for t in range(steps + 1):
                x = int(x0 + (x1 - x0) * t / steps)
                y = int(y0 + (y1 - y0) * t / steps)
                if 0 <= x < image_size and 0 <= y < image_size:
                    image[y, x] = 1.0
                    # Add some thickness
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < image_size and 0 <= ny < image_size:
                                image[ny, nx] = max(image[ny, nx], 0.5)
        
        return torch.from_numpy(image).unsqueeze(0)  # Add channel dim
    
    def compute_embedding(self, route_data: Dict) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for a single route.
        
        Args:
            route_data: Route data dict with geometry and metadata
            
        Returns:
            Dict with 'overall' and per-feature embeddings as numpy arrays
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        with torch.no_grad():
            # Prepare inputs
            inputs = self._prepare_route_tensor(route_data)
            
            # Extract path image separately (model expects img and features)
            path_image = inputs.pop('path_image')
            
            # Get embeddings using return_all=True
            fused, cnn_embed, metadata_embed, individual_embeddings = self.model(
                path_image, inputs, return_all=True
            )
            
            # Convert to numpy
            result = {
                'overall': fused.cpu().numpy().squeeze(),
                'shape': cnn_embed.cpu().numpy().squeeze(),
                'metadata': metadata_embed.cpu().numpy().squeeze()
            }
            
            for name, emb in individual_embeddings.items():
                result[name] = emb.cpu().numpy().squeeze()
            
            return result
    
    def compute_similarity(
        self, 
        route_data: Dict, 
        precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict:
        """
        Compute similarity between a new route and pre-computed embeddings.
        
        Args:
            route_data: New route data
            precomputed_embeddings: Dict of embedding type -> (N, dim) arrays
                                   If None, loads from export directory
        
        Returns:
            Dict with similarity results for each embedding type
        """
        # Compute embedding for new route
        new_embeddings = self.compute_embedding(route_data)
        
        # Load pre-computed embeddings if not provided
        if precomputed_embeddings is None:
            precomputed_embeddings = self._load_precomputed_embeddings()
        
        results = {}
        
        for emb_type, new_emb in new_embeddings.items():
            if emb_type not in precomputed_embeddings:
                continue
            
            stored_embs = precomputed_embeddings[emb_type]
            
            # Compute cosine similarity
            new_norm = new_emb / (np.linalg.norm(new_emb) + 1e-8)
            stored_norms = stored_embs / (np.linalg.norm(stored_embs, axis=1, keepdims=True) + 1e-8)
            
            similarities = np.dot(stored_norms, new_norm)
            
            results[emb_type] = similarities.tolist()
        
        return results
    
    def find_similar_routes(
        self, 
        route_data: Dict, 
        top_k: int = 10,
        embedding_type: str = 'overall'
    ) -> List[Dict]:
        """
        Find most similar routes to a new route.
        
        Args:
            route_data: New route data
            top_k: Number of similar routes to return
            embedding_type: Which embedding to use for similarity
        
        Returns:
            List of dicts with route index, id, and similarity scores
        """
        # Load routes metadata
        routes_path = self.export_dir / "routes.json"
        with open(routes_path) as f:
            metadata = json.load(f)
        
        routes = metadata['routes']
        
        # Compute all similarities
        all_similarities = self.compute_similarity(route_data)
        
        # Use 'overall' if available, otherwise use first available type
        if embedding_type not in all_similarities:
            if 'overall' in all_similarities:
                embedding_type = 'overall'
            else:
                embedding_type = list(all_similarities.keys())[0]
        
        similarities = all_similarities[embedding_type]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            route_meta = routes[idx].copy()
            
            # Add all similarity scores
            route_meta['similarities'] = {
                emb_type: all_similarities[emb_type][idx]
                for emb_type in all_similarities
            }
            route_meta['overall_similarity'] = similarities[idx]
            
            results.append(route_meta)
        
        return results
    
    def _load_precomputed_embeddings(self) -> Dict[str, np.ndarray]:
        """Load pre-computed embeddings from binary files."""
        # Load metadata to get embedding info
        routes_path = self.export_dir / "routes.json"
        with open(routes_path) as f:
            metadata = json.load(f)
        
        embeddings = {}
        n_routes = metadata['count']
        emb_dim = metadata.get('embedding_dim', 128)
        
        # Try new format first (per-feature embedding files)
        embedding_types = metadata.get('embedding_types', [])
        if embedding_types:
            for emb_type in embedding_types:
                emb_path = self.export_dir / f"embeddings_{emb_type}.bin"
                if emb_path.exists():
                    data = np.fromfile(emb_path, dtype=np.float32)
                    emb_dim_actual = len(data) // n_routes
                    embeddings[emb_type] = data.reshape(n_routes, emb_dim_actual)
        else:
            # Old format: single embeddings.bin file (overall only)
            emb_path = self.export_dir / "embeddings.bin"
            if emb_path.exists():
                data = np.fromfile(emb_path, dtype=np.float32)
                embeddings['overall'] = data.reshape(n_routes, emb_dim)
        
        return embeddings


# ============================================================================
# HTTP Service (optional - requires FastAPI)
# ============================================================================

def create_app():
    """Create FastAPI app for HTTP service."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import Any
    except ImportError:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None
    
    app = FastAPI(title="Route Inference Service", version="1.0.0")
    service = RouteInferenceService()
    
    class RouteInput(BaseModel):
        route_data: Dict[str, Any]
    
    class SimilarRoutesRequest(BaseModel):
        route_data: Dict[str, Any]
        top_k: int = 10
        embedding_type: str = 'overall'
    
    @app.get("/health")
    def health():
        return {"status": "healthy", "model_loaded": service.model is not None}
    
    @app.post("/embedding")
    def compute_embedding(req: RouteInput):
        """Compute embedding for a single route."""
        try:
            embeddings = service.compute_embedding(req.route_data)
            # Convert numpy arrays to lists for JSON serialization
            return {k: v.tolist() for k, v in embeddings.items()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/similar")
    def find_similar(req: SimilarRoutesRequest):
        """Find similar routes to a new route."""
        try:
            results = service.find_similar_routes(
                req.route_data, 
                top_k=req.top_k,
                embedding_type=req.embedding_type
            )
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/similarity")
    def compute_similarity(req: RouteInput):
        """Compute similarity to all routes in the index."""
        try:
            similarities = service.compute_similarity(req.route_data)
            return similarities
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    return app


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Route Inference Service")
    parser.add_argument("--serve", action="store_true", help="Run as HTTP service")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host (default: 0.0.0.0)")
    parser.add_argument("--route", type=str, help="Path to route JSON file for single inference")
    parser.add_argument("--top-k", type=int, default=10, help="Number of similar routes to return")
    parser.add_argument("--export-dir", default="export_kotlin", help="Export directory path")
    
    args = parser.parse_args()
    
    if args.serve:
        # Run HTTP service
        app = create_app()
        if app is None:
            sys.exit(1)
        
        import uvicorn
        print(f"🚀 Starting inference service on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.route:
        # Single route inference
        service = RouteInferenceService(export_dir=args.export_dir)
        
        with open(args.route) as f:
            route_data = json.load(f)
        
        print(f"\n🔍 Finding {args.top_k} similar routes...")
        results = service.find_similar_routes(route_data, top_k=args.top_k)
        
        print(f"\n📊 Top {args.top_k} similar routes:")
        print("-" * 60)
        
        for i, route in enumerate(results):
            route_id = route.get('id', f"Route {route.get('index')}")
            print(f"\n{i+1}. {route_id}")
            print(f"   Overall similarity: {route['overall_similarity']:.4f}")
            print("   Feature similarities:")
            for feat, sim in route['similarities'].items():
                if feat != 'overall':
                    print(f"     - {feat}: {sim:.4f}")
        
        # Also output as JSON for programmatic use
        print("\n\n📤 JSON output:")
        print(json.dumps(results, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
