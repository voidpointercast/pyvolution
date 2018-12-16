from typing import Callable, TypeVar, Generic, Optional, Mapping, Sequence, Union, MutableMapping, Tuple, Iterable, MutableSequence
from operator import add, sub, mul, truediv
from itertools import chain, cycle
from attr import attrs, attrib, Factory

T = TypeVar('T')
Instruction = TypeVar('Instruction')
TargetHandler = Callable[[int, Optional[T]], T]
Profile = TypeVar('Profile')
Machine = Callable[[Sequence[Instruction]],Profile]


@attrs
class Storage(Generic[T]):
    storage: Generic[T] = attrib()
    read: Callable[[int], T] = attrib()
    write: Callable[[int, T], None] = attrib()
    positions: Sequence[int] = attrib()
    name: Optional[str] = attrib(default=None)


Handler = Callable[[Mapping[int, Storage], Instruction, int], int]


@attrs
class InstructionSetArchitecture(Generic[Instruction]):
    get_opcode: Callable[[Instruction], int] = attrib()
    handler_map: Mapping[int, Handler] = attrib()

    def add_operation(self, handler: Handler):
        self.handler_map[max(self.handler_map.keys()) + 1] = handler



def create_random_access_storage(size: int, default: T, name: Optional[str]=None) -> Storage[T]:
    storage = dict(zip(range(size), cycle([default])))
    return Storage(
        storage,
        storage.__getitem__,
        storage.__setitem__,
        tuple(storage.keys()),
        name
    )


def create_stack_storage(name: Optional[str]=None) -> Storage[T]:
    stack = list()
    return Storage(stack, stack.pop, stack.append, tuple(range(len(stack))), name)


def create_storage_targets(storages: Sequence[Storage]) -> Mapping[int, Storage]:
    return dict(
        (i, storage)
        for(i, storage) in enumerate(storages)
    )


@attrs
class SimpleRISCInstruction:
    opcode: int = attrib()
    left: Tuple[int, int] = attrib()
    right: Tuple[int, int] = attrib()
    target: Tuple[int, int] = attrib()

    def get_opcode(self) -> int:
        return self.opcode



def create_loop_instructions() -> Handler:
    loop_condition_map = dict()
    def loop_handler(storage, instruction, pc) -> int:
        if pc not in loop_condition_map:
            loop_condition_map[pc] = storage[instruction.left[0]].read(instruction.left[1])
        if loop_condition_map[pc]:
            loop_condition_map[pc] -= 1
            return storage[instruction.target[0]].read(instruction.target[1])
        else:
            del loop_condition_map[pc]
            return pc + 1
    return loop_handler


def create_simple_risc_isa(operations: Callable[[T, T], T]) -> InstructionSetArchitecture[SimpleRISCInstruction]:
    """
    :param operations:
    :return:
    >>> create_simple_risc_isa([add, sub, mul])
    """
    def create_handler(operation: Callable[[T, T], T]) -> Handler:
        def handle(storage_map: Mapping[int, Storage], instruction: SimpleRISCInstruction, pc: int) -> int:
            left_type, left_index = instruction.left
            right_type, right_index = instruction.right
            target_type, target_index = instruction.target

            storage_map[target_type].write(
                target_index,
                operation(storage_map[left_type].read(left_index), storage_map[right_type].read(right_index))
            )
            return pc + 1
        return handle


    return InstructionSetArchitecture(
        SimpleRISCInstruction.get_opcode,
        dict(enumerate(map(create_handler, operations)))
    )


def create_machine(
    isa: InstructionSetArchitecture,
    profiler: Callable[[int, Optional[Profile]], Profile],
    storages: Sequence[Storage],
) -> Machine:
    """
    :param get_opcode:
    :param operations:
    :param profiler:
    :param storages:
    :return:
    >>> storage = [create_random_access_storage(5, 1, 'reg')]
    >>> storage[0].write(3, 5)
    >>> storage[0].write(4, 0)
    >>> isa = create_simple_risc_isa([add, sub, mul])
    >>> isa.add_operation(create_loop_instructions())
    >>> def profiler(pc, profile):
    ...     profile = profile if profile else []
    ...     profile.append(dict(pc=pc, storage=dict(storage[0].storage)))
    ...     return profile
    >>> run_programm = create_machine(isa, profiler, storage)
    >>> run_programm(
    ...     [
    ...         SimpleRISCInstruction(0, (0, 0), (0, 1), (0, 0)),
    ...         SimpleRISCInstruction(3, (0, 3), (0, 0), (0, 4))
    ...     ]
    ... )
    {0: {0: 1, 1: 1, 2: 2}}
    """
    storage_mapping = create_storage_targets(storages)

    def run_machine(program: Sequence[Instruction]) -> Profile:
        profile: Profile = None
        pc: int = 0
        while pc < len(program):
            profile = profiler(pc, profile)
            instruction = program[pc]
            handler = isa.handler_map[isa.get_opcode(instruction)]
            pc = handler(storage_mapping, instruction, pc)
        return profile
    return run_machine

