default_settings = {
    # Encoder settings
    "encoder_layers": [32],  # Sequence[int]
    "extrinsic_latent_size": 2,  # int
    "intrinsic_latent_size": 3,  # int
    # Decoder settings
    "decoder_layers": [32],  # Sequence[int]
    "sed_min": 1e3,  # float
    "sed_max": 11e3,  # float
    "sed_bins": 1000,  # int
    "sed_unit": "mag",  # str, "mag" or "flambda"
    # VAE settings
    "batch_norm": True,  # bool
    # Physics Layer settings
    "bandpasses": [  # Sequence[str]
        "lsstu",
        "lsstg",
        "lsstr",
        "lssti",
        "lsstz",
        "lssty",
    ],
    "band_oversampling": 51,  # int
}
