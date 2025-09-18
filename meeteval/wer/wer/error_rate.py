import dataclasses

__all__ = ['ErrorRate', 'combine_error_rates', 'SelfOverlap']

from typing import Optional, Any
import logging
import abc

logger = logging.getLogger('error_rate')


@dataclasses.dataclass(frozen=True)
class SelfOverlap:
    """
    This class represents the self-overlap of a reference or a hypothesis
    """
    overlap_rate: float = dataclasses.field(init=False)

    overlap_time: float
    total_time: float

    @classmethod
    def zero(cls):
        """
        The "neutral element" for error rates.
        Useful as a starting point in sum.
        """
        return SelfOverlap(0, 0)

    def __post_init__(self):
        if self.overlap_time < 0:
            raise ValueError()
        if self.total_time < 0:
            raise ValueError()

        object.__setattr__(
            self, 'overlap_rate',
            self.overlap_time / self.total_time if self.total_time > 0 else 0
        )

    def __add__(self, other: 'SelfOverlap') -> 'SelfOverlap':
        """Combines two error rates"""
        if not isinstance(other, SelfOverlap):
            return NotImplemented
        return SelfOverlap(
            self.overlap_time + other.overlap_time,
            self.total_time + other.total_time,
        )

    def __radd__(self, other: 'int') -> 'SelfOverlap':
        if isinstance(other, int) and other == 0:
            # Special case to support sum.
            return self
        return NotImplemented

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d['overlap_time'], d['total_time'])

    def warn(self, name):
        if self.overlap_rate > 0:
            # TODO: what level? Is this the correct place?
            logger.warning(
                f'Self-overlap detected in {name}. Total overlap: '
                f'{self.overlap_time:.2f} of {self.total_time:.2f} '
                f'({self.overlap_rate * 100:.2f}%).'
            )


class BaseErrorRate(abc.ABC):
    @abc.abstractclassmethod
    def zero(cls):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def asdict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __add__(self, other: 'ErrorRate') -> 'ErrorRate':
        raise NotImplementedError()

    @abc.abstractmethod
    def asdict(self) -> dict:
        """
        Returns a dictionary representation. Used for dumping into json or 
        yaml files.
        """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def from_dict(self, d: dict):
        """
        Constructs an error rate object from a dict. Used to load data from 
        json or yaml files.
        """
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True, repr=False)
class ErrorRate(BaseErrorRate):
    """
    This class represents an error rate. It bundles statistics over the errors
    and makes sure that no wrong arithmetic operations can be performed on
    error rates (e.g., a simple mean).

    This class is frozen because an error rate should not change after it
    has been computed.
    """
    identifier = 'error-rate'
    
    error_rate: float = dataclasses.field(init=False)

    errors: int
    length: int

    insertions: int
    deletions: int
    substitutions: int

    reference_self_overlap: Optional[SelfOverlap]
    hypothesis_self_overlap: Optional[SelfOverlap]

    @classmethod
    def zero(cls):
        """
        The "neutral element" for error rates.
        Useful as a starting point in sum.
        """
        return ErrorRate(0, 0, 0, 0, 0, None, None)

    def __post_init__(self):
        if self.errors < 0:
            raise ValueError()
        if self.length < 0:
            raise ValueError()

        if self.errors == 0:
            error_rate = 0
        elif self.length == 0:
            error_rate = None
        else:
            error_rate = self.errors / self.length
        # We have to use object.__setattr__ in frozen dataclass.
        # The alternative would be a property named `error_rate` and a custom
        # repr
        object.__setattr__(self, 'error_rate', error_rate)
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

        # Only allow add between the same type of error or with the base
        if not isinstance(other, self.__class__) and type(other) is not ErrorRate:
            return NotImplemented
        
        # Return the base class here. Meta information can become
        # meaningless and should be handled in subclasses
        return ErrorRate(
            self.errors + other.errors,
            self.length + other.length,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
            substitutions=self.substitutions + other.substitutions,
            reference_self_overlap=self.reference_self_overlap + other.reference_self_overlap if self.reference_self_overlap is not None and other.reference_self_overlap is not None else None,
            hypothesis_self_overlap=self.hypothesis_self_overlap + other.hypothesis_self_overlap if self.hypothesis_self_overlap is not None and other.hypothesis_self_overlap is not None else None,
        )

    @classmethod
    def _from_dict(cls, d: dict):
        """
        Instantiates `cls` from `d`. Keys in `d` must match the required data 
        for `cls`.

        Used by `from_dict` after identifying and checking the arguments.
        """
        def _get_self_overlap(so):
            if so is None:
                return None
            return SelfOverlap.from_dict(so)

        d = {
            'insertions': None,
            'deletions': None,
            'substitutions': None,
            **d,
            'reference_self_overlap': _get_self_overlap(d.get('reference_self_overlap')),
            'hypothesis_self_overlap': _get_self_overlap(d.get('hypothesis_self_overlap'))
        }
        d.pop('error_rate', None) # Is computed from errors and length
        d.pop('type', None)   # Is needed for automatic identification
        return cls(**d)

    @classmethod
    def from_dict(cls, d: dict):
        """
        >>> ErrorRate.from_dict(ErrorRate(1, 1, 0, 0, 1, None, None).asdict())
        ErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=0, substitutions=1)
        >>> from meeteval.wer.wer.cp import CPErrorRate
        >>> ErrorRate.from_dict(CPErrorRate(1, 1, 0, 0, 1, None, None, 1, 1, 1).asdict())
        CPErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=0, substitutions=1, missed_speaker=1, falarm_speaker=1, scored_speaker=1)
        >>> from meeteval.wer.wer.orc import OrcErrorRate
        >>> ErrorRate.from_dict(OrcErrorRate(1, 1, 0, 0, 1, None, None, (0, 1)).asdict())
        OrcErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=0, substitutions=1, assignment=(0, 1))
        >>> from meeteval.wer.wer.mimo import MimoErrorRate
        >>> ErrorRate.from_dict(MimoErrorRate(1, 1, 0, 0, 1, None, None, [(0, 1)]).asdict())
        MimoErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=0, substitutions=1, assignment=[(0, 1)])
        >>> ErrorRate.from_dict(ErrorRate(1, 1, 0, 0, 1, SelfOverlap(10, 100), SelfOverlap(0, 90)).asdict())
        ErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=0, substitutions=1, reference_self_overlap=SelfOverlap(overlap_rate=0.1, overlap_time=10, total_time=100), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=90))
        """
        if cls is ErrorRate:
            type_ = _guess_type(d)

            from meeteval.wer.wer.cp import CPErrorRate
            from meeteval.wer.wer.orc import OrcErrorRate
            from meeteval.wer.wer.mimo import MimoErrorRate
            from meeteval.wer.wer.di_cp import DICPErrorRate
            from meeteval.der.md_eval import DiaErrorRate

            if type_ == cls.identifier:
                return ErrorRate._from_dict(d)
            elif type_ == CPErrorRate.identifier:
                return CPErrorRate._from_dict(d)
            elif type_ == OrcErrorRate.identifier:
                return OrcErrorRate._from_dict(d)
            elif type_ == MimoErrorRate.identifier:
                return MimoErrorRate._from_dict(d)
            elif type_ == DICPErrorRate.identifier:
                return DICPErrorRate._from_dict(d)
            elif type_ == DiaErrorRate.identifier:
                return DiaErrorRate.from_dict(d)
            else:
                raise AssertionError(f'Error while detecting ErrorRate type.')
        else:
            return cls._from_dict(d)
            

    def __repr__(self):
        return (
                self.__class__.__qualname__ + '(' +
                ', '.join([
                    f"{f.name}={getattr(self, f.name)!r}"
                    for f in dataclasses.fields(self)
                    if getattr(self, f.name) is not None
                ]) + ')'
        )

    def asdict(self):
        d = dataclasses.asdict(self)
        d['type'] = self.identifier
        return d


def combine_error_rates(*error_rates: ErrorRate) -> ErrorRate:
    """
    >>> combine_error_rates(ErrorRate(10, 10, 0, 0, 10, None, None), ErrorRate(0, 10, 0, 0, 0, None, None))
    ErrorRate(error_rate=0.5, errors=10, length=20, insertions=0, deletions=0, substitutions=10)
    >>> combine_error_rates(ErrorRate(10, 10, 0, 0, 10, None, None))
    ErrorRate(error_rate=1.0, errors=10, length=10, insertions=0, deletions=0, substitutions=10)
    >>> combine_error_rates(*([ErrorRate(10, 10, 0, 0, 10, None, None)]*10))
    ErrorRate(error_rate=1.0, errors=100, length=100, insertions=0, deletions=0, substitutions=100)
    >>> combine_error_rates()
    ErrorRate(errors=0, length=0, insertions=0, deletions=0, substitutions=0)
    """
    if isinstance(error_rates, (tuple, list)) and len(error_rates) == 1:
        if dataclasses.is_dataclass(error_rates[0]):
            return error_rates[0]
        error_rates, = error_rates
    if isinstance(error_rates, dict):
        error_rates = error_rates.values()
    assert all([dataclasses.is_dataclass(er) for er in error_rates]), ([dataclasses.is_dataclass(er) for er in error_rates], error_rates)

    # Sum returns an int when the operand is empty.
    # sum(..., start=...) is supported only from Python 3.8+
    if len(error_rates) == 0:
        return ErrorRate.zero()
    return sum(error_rates)


@dataclasses.dataclass(frozen=True)
class CombinedErrorRate(ErrorRate):
    details: 'dict[Any, ErrorRate]'

    @classmethod
    def from_error_rates(cls, error_rates: 'dict[Any, ErrorRate]'):
        from meeteval.wer.utils import _values
        er = sum(_values(error_rates))
        return cls(
            errors=er.errors,
            length=er.length,
            insertions=er.insertions,
            deletions=er.deletions,
            substitutions=er.substitutions,
            reference_self_overlap=er.reference_self_overlap,
            hypothesis_self_overlap=er.hypothesis_self_overlap,
            details=error_rates,
        )

    def __repr__(self):
        return (
                self.__class__.__qualname__ + '(' +
                ', '.join([
                    f"{f.name}={getattr(self, f.name)!r}"
                    if f.name != 'details' else 'details=...'
                    for f in dataclasses.fields(self)
                    if getattr(self, f.name) is not None
                ]) + ')'
        )

def _guess_type(error_rate_dict: dict) -> str:
    """
    Guess the error rate type from an error rate dict.

    Mainly for backwards compatibility for files that do not have the 'type' 
    field set.
    """
    if 'type' in error_rate_dict:
        return error_rate_dict['type']

    # Add keys with defaults for backwards compatibility
    keys = set(error_rate_dict.keys()) | {
        'insertions',
        'deletions',
        'substitutions',
        'reference_self_overlap',
        'hypothesis_self_overlap',
    }

    # Every error rate must have these keys
    required_keys = {
        'errors', 'length', 'error_rate',
        'insertions', 'deletions', 'substitutions',
        'reference_self_overlap', 'hypothesis_self_overlap'
    }

    if keys == required_keys:
        return ErrorRate.identifier

    if keys == required_keys | {
        'missed_speaker', 'falarm_speaker', 'scored_speaker', 'assignment'
    }:
        from meeteval.wer.wer.cp import CPErrorRate
        return CPErrorRate.identifier

    if keys == required_keys | {'assignment'}:
        if isinstance(error_rate_dict['assignment'][0], (tuple, list)):
            from meeteval.wer.wer.mimo import MimoErrorRate
            return MimoErrorRate.identifier
        else:
            from meeteval.wer.wer.orc import OrcErrorRate
            return OrcErrorRate.identifier

    raise ValueError(
        f'Cannot identify error rate type from dict: {keys}', 
        error_rate_dict,
    )
