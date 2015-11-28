

class AcceptanceCone:
    """
    Base class for defining the acceptance cone for rays launched by an observer.
    """
    pass


class SingleRay(AcceptanceCone):
    """
    Fires a single ray along the observer axis. Effectively a delta function acceptance cone.
    """
    pass


class LightCone(AcceptanceCone):
    """
    A conical ray acceptance volume. An example would be the light cone accepted by an optical fibre.
    """


class Hemisphere(AcceptanceCone):
    """
    Samples rays over hemisphere in direction of surface normal.
    """
    pass
