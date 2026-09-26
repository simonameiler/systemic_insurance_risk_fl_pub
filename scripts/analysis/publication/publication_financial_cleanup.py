"""Publication-only numerical cleanup; archived simulation outputs stay unchanged."""

FIGA_ZERO_TOLERANCE_USD = 0.01


def clean_figa_roundoff(frame):
    """Treat sub-cent FIGA residuals as zero before any derived calculation.

    A residual of exactly one cent is retained. Missing values are left to the
    existing loader's zero-event handling and validation.
    """
    column = 'figa_residual_deficit_usd'
    residual = frame[column]
    frame.loc[residual.abs() < FIGA_ZERO_TOLERANCE_USD, column] = 0.0
    return frame
