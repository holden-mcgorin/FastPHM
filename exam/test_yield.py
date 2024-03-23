from typing import Generator


class DataLoader:
    def __init__(self):
        self.dataset = [1, 2, 3, 4, 5]

    def data(self) -> Generator[int, None, None]:
        for i in self.dataset:
            yield i


data = DataLoader().data()
print(data)
print(type(data))
print(next(data))