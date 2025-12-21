# 📚 Guide: Adding New Features to the Modular Route Model

This guide explains how to add new features (like elevation, weather, surface type, etc.) to the route similarity model using the modular architecture.

## 🎯 Overview

The modular architecture uses **abstract base classes** to make adding features extremely easy. You only need to:
1. Choose the right base class (Continuous, Categorical, or MultiValue)
2. Create a one-line encoder subclass
3. Extract the feature from your data
4. Add it to the model initialization

Everything else (fusion, training, similarity, normalization) works automatically!

---

## 🏗️ Available Base Classes

Three base classes handle all the boilerplate:

| Base Class | Use For | Input Shape | Examples |
|-----------|---------|-------------|----------|
| `ContinuousFeatureEncoder` | Single numbers | `(batch, 1)` | pace, distance, elevation, temp, heart rate |
| `CategoricalFeatureEncoder` | Categories (one-hot) | `(batch, num_cat)` | terrain, weather, surface, difficulty |
| `MultiValueFeatureEncoder` | Multiple numbers | `(batch, num_vals)` | time (sin,cos), coordinates, wind |

---

## 📋 Step-by-Step Guide

### **Step 1: Create a New Encoder (Super Simple!)**

Open `modular_features.py` and add ONE LINE of code:

#### Option A: Continuous Feature (Most Common)
```python
# filepath: modular_features.py

# For single scalar values: pace, distance, elevation, temperature, speed, etc.
class ElevationEncoder(ContinuousFeatureEncoder):
    """Encoder for elevation gain (meters)"""
    pass  # That's it! Inherits everything from base class
```

#### Option B: Categorical Feature
```python
# filepath: modular_features.py

# For categories: terrain, weather, surface type, difficulty, etc.
class WeatherEncoder(CategoricalFeatureEncoder):
    """Encoder for weather condition"""
    def __init__(self, embedding_dim=32):
        super(WeatherEncoder, self).__init__(
            num_categories=5,  # sunny, cloudy, rainy, snowy, foggy
            embedding_dim=embedding_dim
        )
```

#### Option C: Multi-Value Feature
```python
# filepath: modular_features.py

# For multiple values: coordinates (lat, lon), wind (speed, direction), etc.
class WindEncoder(MultiValueFeatureEncoder):
    """Encoder for wind (speed, direction_sin, direction_cos)"""
    def __init__(self, embedding_dim=32):
        super(WindEncoder, self).__init__(
            num_values=3,  # 3 input values
            embedding_dim=embedding_dim
        )
```

#### Option D: Custom Architecture (Advanced)
```python
# filepath: modular_features.py

# Customize hidden layer size and dropout if needed
class HeartRateEncoder(ContinuousFeatureEncoder):
    """Encoder for heart rate with deeper network"""
    def __init__(self, embedding_dim=32):
        super(HeartRateEncoder, self).__init__(
            embedding_dim=embedding_dim,
            hidden_dim=128,  # Larger hidden layer (default: 64)
            dropout=0.3       # More dropout (default: 0.2)
        )
```

**🎉 No need to write:**
- ✅ `forward()` method
- ✅ Shape handling logic
- ✅ L2 normalization
- ✅ BatchNorm, Dropout, ReLU layers

**Everything is inherited from the base class!**

---

### **Step 2: Extract Feature from Data**

In your notebook, modify the `HybridDataset` class to extract your new feature:

#### 2a. Add Feature Extraction Function (if needed)

```python
def extract_your_feature(route):
    """
    Extract your feature from the route data.
    
    Args:
        route: Dictionary containing route data
    
    Returns:
        Processed feature value (float or category)
    """
    # Example: Extract elevation gain
    elevation = route.get('ElevationGain', 0)
    return elevation / 100.0  # Normalize to ~100s of meters
    
    # Example: Extract categorical feature
    # return route.get('SurfaceType', 'unknown')
```

#### 2b. Update Dataset `__init__` Method

Find the `HybridDataset.__init__` method and add your feature to the route dictionary:

```python
# In HybridDataset.__init__, around line 1370
self.routes.append({
    'img': img,
    'pace': route['PaceMinPerKm'],
    'terrain': terrain,
    'distance': route['Distance'] / 1000.0,
    'time_sin': time_feat['time_sin'],
    'time_cos': time_feat['time_cos'],
    'time_slot': time_feat['time_slot'],
    'your_feature': extract_your_feature(route)  # ADD THIS LINE
})
```

#### 2c. Update Dataset `__getitem__` Method

Add your feature to both `features1` and `features2` dictionaries:

```python
# In HybridDataset.__getitem__, around line 1395
features1 = {
    'pace': torch.tensor([route1['pace']], dtype=torch.float32),
    'terrain': self.terrain_to_onehot(route1['terrain']),
    'distance': torch.tensor([route1['distance']], dtype=torch.float32),
    'time': torch.tensor([route1['time_sin'], route1['time_cos']], dtype=torch.float32),
    'your_feature': torch.tensor([route1['your_feature']], dtype=torch.float32)  # ADD THIS
}

# Do the same for features2
features2 = {
    'pace': torch.tensor([route2['pace']], dtype=torch.float32),
    'terrain': self.terrain_to_onehot(route2['terrain']),
    'distance': torch.tensor([route2['distance']], dtype=torch.float32),
    'time': torch.tensor([route2['time_sin'], route2['time_cos']], dtype=torch.float32),
    'your_feature': torch.tensor([route2['your_feature']], dtype=torch.float32)  # AND THIS
}
```

---

### **Step 3: Add Encoder to Model**

In the training cell (around line 1420), add your encoder to the `encoders` dictionary:

```python
# In the training cell
encoders = {
    'pace': PaceEncoder(embedding_dim=32),
    'terrain': TerrainEncoder(num_terrains=9, embedding_dim=32),
    'distance': DistanceEncoder(embedding_dim=32),
    'time': TimeEncoder(embedding_dim=32),
    'your_feature': YourFeatureEncoder(embedding_dim=32),  # ADD THIS LINE
}
```

**⚠️ Important:** Make sure to reload the module if you edited `modular_features.py`:

```python
import importlib
import modular_features
importlib.reload(modular_features)
```

---

### **Step 4: Update Visualization (Optional)**

To display your feature in the similarity results, update the `format_feature_info` function:

```python
# In the test cell, around line 1490
def format_feature_info(route):
    """Format route feature values as a readable string"""
    time_hour = int(np.arctan2(route['time_sin'], route['time_cos']) / (2 * np.pi) * 24)
    if time_hour < 0:
        time_hour += 24
    return {
        'terrain': route['terrain'].title(),
        'pace': f"{route['pace']:.1f} min/km",
        'distance': f"{route['distance']:.1f} km",
        'time': f"~{time_hour:02d}:00",
        'your_feature': f"{route['your_feature']:.1f} units"  # ADD THIS LINE
    }
```

Then update the visualization title to include it:

```python
# Around line 1510
title_lines = [
    f'Match #{i}: Route #{similar_idx} (Overall: {sim_overall:.0f}%)',
    f"Terrain: {match_info['terrain']} ({similarities['terrain'][similar_idx]:.0f}%)",
    f"Pace: {match_info['pace']} ({similarities['pace'][similar_idx]:.0f}%)",
    f"Dist: {match_info['distance']} ({similarities['distance'][similar_idx]:.0f}%)",
    f"Time: {match_info['time']} ({similarities['time'][similar_idx]:.0f}%)",
    f"YourFeature: {match_info['your_feature']} ({similarities['your_feature'][similar_idx]:.0f}%)"  # ADD THIS
]
```

---

## 🎉 That's It!

**What happens automatically:**
- ✅ Your feature is encoded to a 32-dim embedding
- ✅ Combined with other features in the fusion layer
- ✅ Used in contrastive loss during training
- ✅ Similarity scores computed for testing
- ✅ Visualization shows the new feature

**No changes needed to:**
- Fusion layer architecture
- Training loop
- Loss function
- Similarity calculation
- Model forward pass

---

## 📝 Complete Example: Adding Elevation

Here's a complete example adding elevation gain as a feature:

### 1. Create Encoder (Just One Line!)

```python
# filepath: modular_features.py
class ElevationEncoder(ContinuousFeatureEncoder):
    """Encoder for elevation gain (in meters)"""
    pass  # That's it!
```

### 2. Extract Feature

```python
# In notebook, before HybridDataset
def extract_elevation(route):
    """Extract and normalize elevation gain"""
    return route.get('ElevationGain', 0) / 100.0  # Normalize to ~100s of meters
```

### 3. Update Dataset

```python
# In HybridDataset.__init__
self.routes.append({
    'img': img,
    'pace': route['PaceMinPerKm'],
    'terrain': terrain,
    'distance': route['Distance'] / 1000.0,
    'time_sin': time_feat['time_sin'],
    'time_cos': time_feat['time_cos'],
    'time_slot': time_feat['time_slot'],
    'elevation': extract_elevation(route)  # NEW
})

# In HybridDataset.__getitem__
features1['elevation'] = torch.tensor([route1['elevation']], dtype=torch.float32)
features2['elevation'] = torch.tensor([route2['elevation']], dtype=torch.float32)
```

### 4. Add to Model

```python
# Reload module first!
import importlib
import modular_features
importlib.reload(modular_features)

# Then update encoders
encoders = {
    'pace': PaceEncoder(embedding_dim=32),
    'terrain': TerrainEncoder(num_terrains=9, embedding_dim=32),
    'distance': DistanceEncoder(embedding_dim=32),
    'time': TimeEncoder(embedding_dim=32),
    'elevation': ElevationEncoder(embedding_dim=32),  # NEW
}
```

### 5. Train & Test!

```python
# Re-run the training cell - that's it!
# Your elevation feature is now part of the model
```

---

## 🚀 Advanced: Adding Multiple Features at Once

You can add multiple features in one go - each is just one line!

```python
# filepath: modular_features.py

# 1. Create multiple encoders (super simple!)
class WeatherEncoder(CategoricalFeatureEncoder):
    """Encoder for weather conditions"""
    def __init__(self, embedding_dim=32):
        super().__init__(num_categories=5, embedding_dim=embedding_dim)

class SurfaceEncoder(CategoricalFeatureEncoder):
    """Encoder for surface type"""
    def __init__(self, embedding_dim=32):
        super().__init__(num_categories=4, embedding_dim=embedding_dim)

class TemperatureEncoder(ContinuousFeatureEncoder):
    """Encoder for temperature"""
    pass

# 2. Extract all features in dataset
self.routes.append({
    # ... existing features ...
    'weather': extract_weather(route),
    'surface': extract_surface(route),
    'temperature': extract_temperature(route)
})

# 3. Add all to model
encoders = {
    # ... existing encoders ...
    'weather': WeatherEncoder(embedding_dim=32),
    'surface': SurfaceEncoder(embedding_dim=32),
    'temperature': TemperatureEncoder(embedding_dim=32),
}
```

---

## 🐛 Troubleshooting

**Error: "KeyError: 'your_feature'"**
- Make sure you added the feature to BOTH `features1` and `features2` in `__getitem__`

**Error: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"**
- Check that your encoder input size matches the feature shape
- Use `.unsqueeze(1)` to convert `(batch,)` to `(batch, 1)`

**Error: "AttributeError: module 'modular_features' has no attribute 'YourEncoder'"**
- You forgot to reload the module! Add `importlib.reload(modular_features)` before training

**Feature similarity always 0%**
- Check that your feature values are not all the same
- Make sure you're normalizing properly (divide by reasonable scale)

---

## 📊 Testing Your New Feature

After adding a feature, test that it works:

```python
# Check that dataset loads correctly
print(f"Sample route features: {hybrid_dataset.routes[0].keys()}")

# Check that encoder produces correct shape
test_encoder = YourFeatureEncoder(embedding_dim=32)
test_input = torch.tensor([[5.0]])
test_output = test_encoder(test_input)
print(f"Encoder output shape: {test_output.shape}")  # Should be (1, 32)

# Check that similarity is computed
print(f"Feature similarities available: {list(similarities.keys())}")
```

---

## 💡 Best Practices

1. **Normalize your features:** Divide by a reasonable scale (e.g., elevation by 100, temperature by 50)
2. **Use consistent embedding_dim:** Stick to 32 for all feature encoders
3. **Choose the right base class:**
   - Single value? → `ContinuousFeatureEncoder`
   - Category? → `CategoricalFeatureEncoder`
   - Multiple values? → `MultiValueFeatureEncoder`
4. **Handle missing data:** Use `.get('feature', default_value)` when extracting
5. **Reload the module:** Always `importlib.reload(modular_features)` after editing
6. **Test incrementally:** Add one feature at a time and verify it works

---

## 🎓 Understanding the Architecture

### How Base Classes Work

```
BaseFeatureEncoder (abstract)
├── Provides: forward(), _handle_input_shape(), L2 normalization
├── Requires: _build_encoder() implementation
│
├── ContinuousFeatureEncoder
│   └── _build_encoder(): 1 → 64 → BN → ReLU → Dropout → embedding_dim
│
├── CategoricalFeatureEncoder  
│   └── _build_encoder(): num_cat → 64 → BN → ReLU → Dropout → embedding_dim
│
└── MultiValueFeatureEncoder
    └── _build_encoder(): num_vals → 64 → BN → ReLU → Dropout → embedding_dim
```

### Feature Flow

```
Route Data → Feature Extraction → Individual Encoders → Fusion → Final Embedding
                                          ↓
                                   32-dim each
                                   (via base class)
                                          ↓
                                   Concatenated
                                          ↓
                                   Fusion Layer
                                          ↓
                                   128-dim final
```

**Benefits:**
- ✅ No boilerplate code - base classes handle everything
- ✅ Consistent architecture across all features
- ✅ Easy to customize (override parameters)
- ✅ Type safety - all encoders have same interface
- ✅ Less bugs - reuse tested code

---

## 📚 Further Reading

- See `modular_features.py` for existing encoder implementations
- Check the notebook cell comments for detailed explanations
- Read the model architecture in `HybridRouteModelModular`

---

**Happy feature engineering! 🚀**

If you have questions, check the troubleshooting section or review existing encoder implementations in `modular_features.py`.