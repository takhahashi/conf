from dataclasses import dataclass

@dataclass
class Address:
    street: str
    city: str
    zip_code: str

@dataclass
class TrainingArgs:
    weight_decay: int
    name: str
    age: int
    address: Address