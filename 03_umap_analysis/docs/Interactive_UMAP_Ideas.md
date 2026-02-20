# Interactive UMAP Visualization Enhancement Ideas

**Date:** 2026-01-29  
**Status:** Planning / Future Development

## Goal
Create dynamic, interactive UMAP visualizations with:
1. **Real-time parameter adjustment** via sliders (n_neighbors, min_dist)
2. **Chemical structure display** on hover/click using SMILES strings
3. **Enhanced interactivity** for exploration and analysis

---

## Key References

### 1. UMAP Examples - Mammoth
**URL:** https://github.com/MNoichl/UMAP-examples-mammoth.git

**Why Relevant:**
- Interactive UMAP implementations with various datasets
- Examples of dynamic parameter exploration
- Best practices for UMAP visualization workflows

**Key Features to Explore:**
- Parameter sweep visualization techniques
- Interactive embedding updates
- Integration with different data types

---

### 2. Plotly Sliders
**URL:** https://plotly.com/python/sliders/

**Why Relevant:**
- Native Python support for interactive controls
- Slider widgets for real-time parameter updates
- Can trigger UMAP re-computation on parameter change

**Key Features to Implement:**
- `n_neighbors` slider (range: 5-100)
- `min_dist` slider (range: 0.0-1.0)
- Conditional sliders for different UMAP types (structural/FPM/descriptor)
- Animation frames for parameter sweeps

**Example Pattern:**
```python
import plotly.graph_objects as go

fig = go.Figure()

# Add slider for n_neighbors
fig.update_layout(
    sliders=[{
        'active': 0,
        'steps': [{
            'args': [{'n_neighbors': n}],
            'label': str(n),
            'method': 'update'
        } for n in range(10, 51, 5)]
    }]
)
```

---

### 3. TMAP - Chemical Space Visualization
**URL:** https://tmap.gdb.tools/?ref=gdb.unibe.ch#simple-graph

**Why Relevant:**
- Specialized for **chemical space visualization**
- Built-in support for **SMILES structure rendering**
- Interactive tree-map layout optimized for molecular data
- Handles large molecular datasets efficiently

**Key Features to Adopt:**
- **Hover tooltips** with 2D structure rendering (RDKit → SVG/PNG)
- **Click events** to display full compound details
- Color coding by properties (MW, cLogP, hit status)
- Search/filter functionality
- Export capabilities

**Chemical Structure Integration:**
```python
from rdkit import Chem
from rdkit.Chem import Draw

def smiles_to_img(smiles):
    """Convert SMILES to image for hover display"""
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(200, 200))
```

---

## Proposed Implementation Roadmap

### Phase 1: Basic Interactive Plotly UMAP
- [x] Static UMAP with Plotly scatter plots
- [ ] Add hover data (SMILES, MW, properties)
- [ ] Color by condition/hit status
- [ ] Export to HTML for sharing

### Phase 2: Parameter Sliders
- [ ] Implement n_neighbors slider with pre-computed embeddings
- [ ] Implement min_dist slider
- [ ] Cache multiple UMAP fits for smooth transitions
- [ ] Show quality metrics (Spearman ρ) for current parameters

### Phase 3: Chemical Structure Display
- [ ] Generate 2D structure images from SMILES (RDKit)
- [ ] Embed structures as base64 in hover tooltips
- [ ] Click event to show large structure + properties panel
- [ ] Highlight substructure matches for hits

### Phase 4: Advanced Interactivity
- [ ] Side-by-side comparison (Structural vs FPM vs Descriptor)
- [ ] Linked brushing across panels
- [ ] Filter by property ranges (MW, cLogP, etc.)
- [ ] Export selected compounds

---

## Technical Stack

### Core Libraries
- **Plotly** (`plotly.graph_objects`, `plotly.express`) - Interactive plotting
- **Dash** (optional) - Web app framework for advanced controls
- **RDKit** - SMILES → 2D structure rendering
- **UMAP** - Dimensionality reduction (existing)
- **Pandas** - Data management

### Structure Rendering Options
1. **RDKit → SVG/PNG:** `Draw.MolToImage()`, `Draw.MolToSVG()`
2. **Base64 encoding:** Embed images directly in HTML
3. **Server-side rendering:** Generate on-demand for large datasets

### Performance Considerations
- **Pre-compute UMAP grid:** Cache embeddings for slider combinations
- **Structure caching:** Pre-render structures for faster display
- **Decimation:** Show subset of points for >10k compounds
- **WebGL:** Use `plotly.graph_objects.Scattergl` for large datasets

---

## Example Workflow

```python
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
import umap
import pandas as pd

# Load data
df = pd.read_csv("umap_results.csv")

# Pre-compute structure images
def get_structure_img(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(150, 150))
    # Convert to base64 for embedding
    import io, base64
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

df['structure_img'] = df['Smiles'].apply(get_structure_img)

# Create interactive scatter
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['UMAP_1'],
    y=df['UMAP_2'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['Total Molweight'],
        colorscale='Viridis',
        showscale=True
    ),
    customdata=df[['Smiles', 'structure_img', 'Total Molweight', 'cLogP']],
    hovertemplate='<b>MW:</b> %{customdata[2]:.1f}<br>' +
                  '<b>cLogP:</b> %{customdata[3]:.2f}<br>' +
                  '<img src="data:image/png;base64,%{customdata[1]}">' +
                  '<extra></extra>'
))

# Add sliders
fig.update_layout(
    sliders=[{
        'active': 3,
        'steps': [{'label': f'n={n}', 'method': 'restyle', 'args': ['n_neighbors', n]} 
                  for n in [10, 15, 20, 30, 40, 50]]
    }],
    title="Interactive UMAP: Macrocycle Chemical Space"
)

fig.write_html("interactive_umap.html")
```

---

## Current Status (2026-01-29)

### Completed:
- ✅ Parameter sweep framework (9 conditions × parameter grid)
- ✅ Quality metrics (Spearman ρ, overlap, purity)
- ✅ Static UMAP visualizations (SVG output)
- ✅ Three-panel analysis (Structural, FPM, Descriptor)

### Next Steps:
1. Review UMAP-examples-mammoth repository for implementation patterns
2. Create Plotly version of existing UMAP plots
3. Implement basic sliders with pre-computed parameter sweep results
4. Add chemical structure hover using RDKit

### Questions to Resolve:
- **Live vs Pre-computed:** Recompute UMAP on slider change (slow) vs pre-cache grid (memory)?
- **Structure resolution:** Thumbnail size for hover? Full size on click?
- **Framework choice:** Pure Plotly HTML or Dash web app?
- **Dataset size limits:** Current ~1400 compounds manageable, but plan for scalability

---

## Resources & Documentation

- UMAP Python API: https://umap-learn.readthedocs.io/
- Plotly Python: https://plotly.com/python/
- RDKit Drawing: https://www.rdkit.org/docs/GettingStartedInPython.html#drawing-molecules
- Dash Tutorial: https://dash.plotly.com/
- TMAP Paper: Probst & Reymond (2020) J. Cheminform. 12:12

---

## Notes

- Consider exporting current parameter sweep results as interactive HTML for immediate exploration
- TMAP uses tree-map projection (different from UMAP) - evaluate if hybrid approach useful
- Structure rendering may be slow for 1400 compounds - consider lazy loading or thumbnails
- Sliders should display quality metrics to guide parameter selection
- Could combine with ChEMBL/PubChem links for literature exploration
