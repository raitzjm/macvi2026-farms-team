# Preprocessing Requirements

**CRITICAL** — training and eval pipelines must match the challenge server exactly.
Mismatches silently degrade server scores.

## Challenge Server Preprocessing

The challenge server preprocesses images before feeding them to our ONNX:
1. Resize with **bilinear interpolation, aspect ratio preserved** → fits within 384×768
2. **Center** the resized image in the 384×768 canvas
3. Fill remaining space with **mirror padding** (not zero padding)
4. Normalize with ImageNet mean/std (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

## Required mmseg Pipeline Settings

**Test pipeline:**
```python
dict(type='Resize', keep_ratio=True)
dict(type='Pad', size=crop_size, padding_mode='reflect')  # with centering
```

**Train pipeline:**
```python
dict(type='Pad', size=crop_size, padding_mode='reflect')
# padding_mode='reflect' is mirror padding; default pad_val=0 is WRONG
```

## Phase-Specific Pipeline Notes

### Phase 2+ (`macvi2` env, mmseg 1.x)

Train pipeline: `RandomResize (keep_ratio=True)` → `RandomCrop` → `ReflectPad`
Test pipeline: `Resize (keep_ratio=True)` → `ReflectPad` → `PackSegInputs`

- Normalization is in `data_preprocessor` (NOT in pipeline — mmseg 1.x change)
- `ReflectPad` is a custom transform registered via `@TRANSFORMS.register_module()`

### Phase 2.5 Fixes Applied

1. `keep_ratio=True` in train `RandomResize` (Phase 2 used `keep_ratio=False`)
2. `img_padding_size` written to results for de-padding during val metric computation
3. Centred val/test padding (`center=True` in `ReflectPad`)
4. Ignore-area overlay (`MacViSegVisualizer` subclass) for WandB prediction images

See `phase2_5/PHASE2_5_NOTES.md` for full details.
