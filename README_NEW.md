# 🎯 Fully Modular Feature System

**Add features by editing ONLY `modular_features.py` - no notebook changes required!**

## Philosophy: Zero Notebook Modifications

This system uses a **Feature Registry** that automatically handles everything:
- ✅ Feature discovery
- ✅ Encoder creation
- ✅ Feature extraction from routes
- ✅ Tensor conversion
- ✅ Model architecture updates
- ✅ Training integration
- ✅ Similarity computation

**When you add a feature to the registry, everything else happens automatically.**

---

## 🚀 Quick Start: Adding a New Feature

Edit **only** `modular_features.py` (4 simple steps):

### Example: Adding Elevation

```python
# filepath: modular_features.py

# 1. Create encoder class (one line!)
class ElevationEncoder(ContinuousFeatureEncoder):
    pass

# 2. Create extraction function
def extract_elevation(route: Dict) -> float:
    """Extract elevation gain from route"""
    return route.get('elevation_gain', 0.0)

# 3. Create conversion function  
def elevation_to_tensor(elevation: float) -> torch.Tensor:
    """Convert elevation to tensor"""
    return torch.tensor([elevation], dtype=torch.float32)

# 4. Register the feature
FeatureRegistry.register('elevation', ElevationEncoder, extract_elevation, elevation_to_tensor)
```

**That's it!** Re-run the notebook and elevation will automatically:
- ✅ Be extracted from routes
- ✅ Get its own encoder
- ✅ Be included in training
- ✅ Show up in similarity results
- ✅ Appear in visualizations

**No notebook code changes needed!**

---

## 🏗️ Base Classes

Choose the appropriate base class for your feature:

| Base Class | Use For | Example |
|-----------|---------|---------|
| `ContinuousFeatureEncoder` | Single numbers | pace, distance, elevation, temperature |
| `CategoricalFeatureEncoder` | Categories | terrain, weather, surface type |
| `MultiValueFeatureEncoder` | Multiple values | time (sin, cos), coordinates |

---

## 📚 More Examples

### Example 1: Weather (Categorical)

```python
# filepath: modular_features.py

WEATHER_TYPES = ['sunny', 'cloudy', 'rainy', 'snowy']

class WeatherEncoder(CategoricalFeatureEncoder):
    def __init__(self, embedding_dim=32):
        super().__init__(num_categories=4, embedding_dim=embedding_dim)

def extract_weather(route: Dict) -> str:
    return route.get('weather', 'sunny')

def weather_to_tensor(weather: str) -> torch.Tensor:
    idx = WEATHER_TYPES.index(weather) if weather in WEATHER_TYPES else 0
    one_hot = torch.zeros(len(WEATHER_TYPES), dtype=torch.float32)
    one_hot[idx] = 1.0
    return one_hot

FeatureRegistry.register('weather', WeatherEncoder, extract_weather, weather_to_tensor)
```

### Example 2: Heart Rate (Continuous)

```python
# filepath: modular_features.py

class HeartRateEncoder(ContinuousFeatureEncoder):
    pass

def extract_heart_rate(route: Dict) -> float:
    return route.get('avg_heart_rate', 140.0)

def heart_rate_to_tensor(hr: float) -> torch.Tensor:
    return torch.tensor([hr], dtype=torch.float32)

FeatureRegistry.register('heart_rate', HeartRateEncoder, extract_heart_rate, heart_rate_to_tensor)
```

### Example 3: Start Location (Multi-Value)

```python
# filepath: modular_features.py

class StartLocationEncoder(MultiValueFeatureEncoder):
    def __init__(self, embedding_dim=32):
        super().__init__(num_values=2, embedding_dim=embedding_dim)  # lat, lon

def extract_start_location(route: Dict) -> Tuple[float, float]:
    geometry = route.get('geometry', [])
    if geometry:
        return geometry[0][1], geometry[0][0]  # lat, lon
    return 0.0, 0.0

def start_location_to_tensor(location: Tuple[float, float]) -> torch.Tensor:
    lat, lon = location
    # Normalize to [-1, 1] range (approximate)
    lat_norm = lat / 90.0
    lon_norm = lon / 180.0
    return torch.tensor([lat_norm, lon_norm], dtype=torch.float32)

FeatureRegistry.register('start_location', StartLocationEncoder, extract_start_location, start_location_to_tensor)
```

---

## 🎓 How It Works

### 1. Feature Registry
The `FeatureRegistry` class stores all feature definitions:
```python
FeatureRegistry.register(
    name='pace',                    # Feature name
    encoder_class=PaceEncoder,       # Encoder class (not instance!)
    extractor=extract_pace,          # Function: route → value
    tensor_converter=pace_to_tensor  # Function: value → tensor
)
```

### 2. Automatic Dataset Integration
`HybridDatasetModular` uses the registry:
```python
# In __getitem__:
features_raw = FeatureRegistry.extract_features(route)  # Call all extractors
features = FeatureRegistry.convert_to_tensors(features_raw)  # Convert all to tensors
```

### 3. Automatic Model Creation
`MetadataEncoderModular` creates encoders automatically:
```python
def __init__(self, fusion_dim=128, embedding_dim=32):
    # Automatically create ALL registered encoders
    self.encoders = nn.ModuleDict(FeatureRegistry.create_encoders(embedding_dim))
```

### 4. Automatic Similarity
The test cell automatically computes similarities for all features:
```python
for feat_name in FeatureRegistry.get_all_features():
    similarities[feat_name] = compute_similarity(embeddings[feat_name])
```

---

## 🔧 Advanced: Custom Architecture

If you need a custom encoder architecture, override `_build_encoder()`:

```python
class ComplexFeatureEncoder(ContinuousFeatureEncoder):
    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(1, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim)
        )
```

---

## ✅ Best Practices

1. **Naming**: Use descriptive names (`pace`, `elevation`, `heart_rate`)
2. **Defaults**: Provide sensible defaults in extractors
3. **Normalization**: Normalize features to similar ranges
4. **Documentation**: Add docstrings to your encoders
5. **Testing**: Run a quick test after adding features

---

## 🐛 Troubleshooting

### Feature not showing up?
- Check `FeatureRegistry.register()` was called
- Verify extractor returns the right type
- Make sure tensor converter returns `torch.Tensor`

### Shape errors?
- Continuous: return `(1,)` tensor
- Categorical: return `(num_categories,)` tensor
- Multi-value: return `(num_values,)` tensor

### Missing data?
- Add default values in extractors
- Handle missing keys with `.get(key, default)`

---

## 📊 Current Features

The system comes with 4 features out of the box:

| Feature | Type | Encoder | Description |
|---------|------|---------|-------------|
| `pace` | Continuous | `PaceEncoder` | Running pace (min/km) |
| `terrain` | Categorical | `TerrainEncoder` | Terrain type (9 categories) |
| `distance` | Continuous | `DistanceEncoder` | Route distance (km) |
| `time` | Multi-Value | `TimeEncoder` | Time of day (sin, cos) |

---

## 🎯 Summary

**To add a feature:**
1. Edit `modular_features.py`
2. Create encoder class (inherit from base)
3. Create `extract_feature()` function
4. Create `feature_to_tensor()` function
5. Call `FeatureRegistry.register()`

**No changes needed in:**
- ❌ Notebook cells
- ❌ Dataset class
- ❌ Training loop
- ❌ Testing code
- ❌ Visualization code

**Everything is automatic!** 🚀
