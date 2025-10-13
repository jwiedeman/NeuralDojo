# Asteroid Game troubleshooting

## Metal headless pipeline validation failure

**Error message**

```
wgpu error: Validation Error
Caused by:
    In Device::create_compute_pipeline
      note: label = `headless-point-cull-pipeline`
    Internal error: MSL: FeatureNotImplemented("atomic CompareExchange")
```

**Why it happens**

The headless point-culling pipeline relies on Metal's
`atomic_compare_exchange` instruction. Apple GPUs only expose this
instruction on macOS 13 (Ventura) and newer. Running the headless
renderer on earlier versions of macOS therefore fails validation when the
compute pipeline is created.

**Workarounds**

1. **Prefer standard presentation.** Launch the binary without a
   `--headless` flag so that the renderer follows the regular presentation
   path, which does not need the missing atomic operation.
2. **Upgrade macOS.** Moving to macOS Ventura (13) or newer unlocks the
   required Metal feature on Apple GPUs.
3. **Use a Linux or Windows host.** Both Vulkan (Linux) and Direct3D12
   (Windows) expose the compare-exchange primitive, so the headless
   renderer works as expected on those platforms.

We will ship a Metal-friendly variant of the compute shader so that the
headless renderer can run everywhere. Until then, stick to one of the
options above if you hit this validation failure.
