import dataclasses

__all__ = ['ErrorRate', 'combine_error_rates']


@dataclasses.dataclass(frozen=True)
class ErrorRate:
    """
    This class represents an error rate. It bundles statistics over the errors
    and makes sure that no wrong arithmetic operations can be performed on
    error rates (e.g., a simple mean).

    This class is frozen because an error rate should not change after it
    has been computed.
    """
    errors: int
    length: int

    insertions: int
    deletions: int
    substitutions: int

    error_rate: int = dataclasses.field(init=False)

    @classmethod
    def zero(cls):
        """
        The "neutral element" for error rates.
        Useful as a starting point in sum.
        """
        return ErrorRate(0, 0, 0, 0, 0)

    def __post_init__(self):
        if self.errors < 0:
            raise ValueError()
        if self.length < 0:
            raise ValueError()

        # We have to use object.__setattr__ in frozen dataclass.
        # The alternative would be a property named `error_rate` and a custom
        # repr
        object.__setattr__(
            self, 'error_rate',
            self.errors / self.length if self.length > 0 else None
        )
        assert self.length == 0 or self.error_rate >= 0
        errors = self.insertions + self.deletions + self.substitutions
        if errors != self.errors:
            raise RuntimeError(
                f'ins {self.insertions} + del {self.deletions} + sub {self.substitutions} != errors {self.errors}')

    def __radd__(self, other: 'int') -> 'ErrorRate':
        if isinstance(other, int) and other == 0:
            # Special case to support sum.
            return self
        return NotImplemented

    def __add__(self, other: 'ErrorRate') -> 'ErrorRate':
        """Combines two error rates"""
        if not isinstance(other, ErrorRate):
            return NotImplemented
        # Return the base class here. Meta information can become
        # meaningless and should be handled in subclasses
        return ErrorRate(
            self.errors + other.errors,
            self.length + other.length,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
            substitutions=self.substitutions + other.substitutions,
        )

    @classmethod
    def from_dict(self, d: dict):
        """
        >>> ErrorRate.from_dict(dataclasses.asdict(ErrorRate(1, 1, 0, 0, 1)))
        ErrorRate(errors=1, length=1, insertions=0, deletions=0, substitutions=1, error_rate=1.0)
        >>> from meeteval.wer.wer.cp import CPErrorRate
        >>> ErrorRate.from_dict(dataclasses.asdict(CPErrorRate(1, 1, 0, 0, 1, 1, 1, 1)))
        CPErrorRate(errors=1, length=1, insertions=0, deletions=0, substitutions=1, error_rate=1.0, missed_speaker=1, falarm_speaker=1, scored_speaker=1, assignment=None)
        >>> from meeteval.wer.wer.orc import OrcErrorRate
        >>> ErrorRate.from_dict(dataclasses.asdict(OrcErrorRate(1, 1, 0, 0, 1, (0, 1))))
        OrcErrorRate(errors=1, length=1, insertions=0, deletions=0, substitutions=1, error_rate=1.0, assignment=(0, 1))
        >>> from meeteval.wer.wer.mimo import MimoErrorRate
        >>> ErrorRate.from_dict(dataclasses.asdict(MimoErrorRate(1, 1, 0, 0, 1, [(0, 1)])))
        MimoErrorRate(errors=1, length=1, insertions=0, deletions=0, substitutions=1, error_rate=1.0, assignment=[(0, 1)])
        """
        # For backward compatibility, set default values.
        d.setdefault('insertions', None)
        d.setdefault('deletions', None)
        d.setdefault('substitutions', None)

        if d.keys() == {
            'errors', 'length', 'error_rate',
            'insertions', 'deletions', 'substitutions',
        }:
            return ErrorRate(
                errors=d['errors'],
                length=d['length'],
                insertions=d['insertions'],
                deletions=d['deletions'],
                substitutions=d['substitutions'],
            )

        if d.keys() == {
            'errors', 'length', 'error_rate',
            'insertions', 'deletions', 'substitutions',
            'missed_speaker', 'falarm_speaker', 'scored_speaker',
            'assignment',
        }:
            from .cp import CPErrorRate
            return CPErrorRate(
                errors=d['errors'], length=d['length'],
                insertions=d['insertions'],
                deletions=d['deletions'],
                substitutions=d['substitutions'],
                missed_speaker=d['missed_speaker'],
                falarm_speaker=d['falarm_speaker'],
                scored_speaker=d['scored_speaker'],
                assignment=d['assignment'],
            )

        if d.keys() == {
            'errors', 'length', 'error_rate',
            'insertions', 'deletions', 'substitutions',
            'assignment',
        }:
            if isinstance(d['assignment'][0], (tuple, list)):
                from .mimo import MimoErrorRate
                XErrorRate = MimoErrorRate
            else:
                from .orc import OrcErrorRate
                XErrorRate = OrcErrorRate

            return XErrorRate(
                errors=d['errors'],
                length=d['length'],
                insertions=d['insertions'],
                deletions=d['deletions'],
                substitutions=d['substitutions'],
                assignment=d['assignment'],
            )
        raise ValueError(d.keys(), d)


def combine_error_rates(*error_rates: ErrorRate) -> ErrorRate:
    """
    >>> combine_error_rates(ErrorRate(10, 10, 0, 0, 10), ErrorRate(0, 10, 0, 0, 0))
    ErrorRate(errors=10, length=20, insertions=0, deletions=0, substitutions=10, error_rate=0.5)
    >>> combine_error_rates(ErrorRate(10, 10, 0, 0, 10))
    ErrorRate(errors=10, length=10, insertions=0, deletions=0, substitutions=10, error_rate=1.0)
    >>> combine_error_rates(*([ErrorRate(10, 10, 0, 0, 10)]*10))
    ErrorRate(errors=100, length=100, insertions=0, deletions=0, substitutions=100, error_rate=1.0)
    """
    if len(error_rates) == 1:
        return error_rates[0]
    return sum(error_rates, error_rates[0].zero())
