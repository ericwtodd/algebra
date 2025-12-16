import random

# Represents an item in a Magma or Quasigroup
class MagmaItem:
    def __init__(self, value, magma):
        self._value = value
        self._magma = magma
    
    @property
    def value(self):
        return self._value
    
    def __mul__(self, other):
        if not isinstance(other, MagmaItem):
            raise TypeError("Can only multiply MagmaItem with another MagmaItem")
        return self._magma.operation(self, other)
    
    def __truediv__(self, other):
        if not isinstance(other, MagmaItem):
            raise TypeError("Can only divide MagmaItem with another MagmaItem")
        return self._magma.right_division(self, other)
    
    def __floordiv__(self, other):
        if not isinstance(other, MagmaItem):
            raise TypeError("Can only divide MagmaItem with another MagmaItem")
        return self._magma.left_division(self, other)
    
    def __repr__(self):
        return f"MagmaItem({self._value})"
    
    def __hash__(self):
        return hash((self._value, id(self._magma)))
    
    def __eq__(self, other):
        return isinstance(other, MagmaItem) and self._value == other._value and id(self._magma) == id(other._magma)


# Represents a generic Magma structure
class Magma:
    def __init__(self, order, seed=None, table=None):
        self._order = order
        self.seed = seed
        self._items = [MagmaItem(i, self) for i in range(order)]
        if table is not None:
            assert len(table) == order
            assert seed is None
            self._operation_table = []
            for row in table:
                assert len(row) == order
                for item in row:
                    assert type(item) == int and 0 <= item < order
                self._operation_table.append([self._items[c] for c in row])
        else:
            rng = random.Random(seed)
            self._operation_table = [[
                rng.choice(self._items) for _ in range(order)] for _ in range(order)]
    
    def generate(self):
        return self._items[:]
    
    def operation(self, a, b):
        if a not in self._items or b not in self._items:
            raise ValueError("Operands must be valid MagmaItems from this Magma")
        return self._operation_table[a.value][b.value]
    
    def __repr__(self):
        return f"Magma(order={self._order}, seed={self.seed})"
    
    def __eq__(self, other):
        return isinstance(other, Magma) and self._operation_table == other._operation_table and self._order == other._order
    
    def operation_table_str(self):
        return '\n'.join(' '.join(str(item.value) for item in row) for row in self._operation_table)
    
    def order(self):
        return self._order

    def is_semigroup(self, verbose=False):
        '''
        Checks assosciativity for every triple.
        '''
        for a in self.generate():
            for b in self.generate():
                ab = a * b
                for c in self.generate():
                    ab_c = ab * c
                    a_bc = a * (b * c)
                    if ab_c != a_bc:
                        if verbose:
                            print(f'{a._value} {b._value} = {ab._value}')
                            print(f'{ab._value} {c._value} = {ab_c._value}')
                            print(f'{b._value} {c._value} = {(b * c)._value}')
                            print(f'{a._value} {(b * c)._value} = {a_bc._value}')
                        return False
        return True

    def has_identity(self):
        return self.find_identity() is not None

    def find_identity(self):
        '''
        Searches for an identity element.
        '''
        for a in self.generate():
            if all(a * b == b for b in self.generate()):
                return a
        return None

    def is_quasigroup(self):
        '''
        Verifies uniqueness of left and right divisors.
        '''
        for a in self.generate():
            if len(set(a * b for b in self.generate())) < self.order():
                return False
        return True

    def is_loop(self):
        return self.has_identity() and self.is_quasigroup()

    def is_monoid(self):
        return self.has_identity() and self.is_semigroup()

    def is_group(self):
        return self.is_loop() and self.is_semigroup()

    def _cache_division_tables(self):
        if hasattr(self, '_left_division_table'):
            return
        left_div = [[None for _ in range(self._order)] for _ in range(self._order)]
        right_div = [[None for _ in range(self._order)] for _ in range(self._order)]
        def update(d, k, v):
            if d[k] is None:
                d[k] = v
            elif type(d[k]) == set:
                d[k].add(v)
            else:
                d[k] = set(d[k], v)
        for a in self._items:
            for b in self._items:
                c = self.operation(a, b)
                update(right_div[c.value], b.value, a)
                update(left_div[a.value], c.value, b)
        self._left_division_table = left_div
        self._right_division_table = right_div

    def left_division(self, a, b):
        self._cache_division_tables()
        return self._left_division_table[a.value][b.value]
      
    def right_division(self, a, b):
        self._cache_division_tables()
        return self._right_division_table[a.value][b.value]

    def division_table_str(self, table):
        return '\n'.join(' '.join(str(item.value if item else '-') for item in row) for row in table)

# Represents a Quasigroup with division operations
class Quasigroup(Magma):
    def __init__(self, order, seed=None):
        super().__init__(order, table=self._generate_latin_square(order))
        # self._operation_table = self._generate_latin_square(order)
    
    def _generate_latin_square(self, order, seed=None):
        # Jacobson, Matthews, 1996. "Generating Uniformly Distributed Random Latin Squares"
        cube = [[[0 for _ in range(order)] for _ in range(order)] for _ in range(order)]
        for i in range(order):
            for j in range(order):
                cube[i][j][(i + j) % order] = 1
        
        is_proper = True
        improper_cell = None
        min_iterations = order ** 3
        step = 0
        rng = random.Random(seed)
        
        while not is_proper or step < min_iterations:
            t = [0, 0, 0]
            c = [0, 0, 0]
            
            if is_proper:
                while True:
                    t = [rng.randint(0, order - 1) for _ in range(3)]
                    if cube[t[0]][t[1]][t[2]] == 0:
                        break
                c[0] = next(i for i in range(order) if cube[i][t[1]][t[2]] == 1)
                c[1] = next(j for j in range(order) if cube[t[0]][j][t[2]] == 1)
                c[2] = next(k for k in range(order) if cube[t[0]][t[1]][k] == 1)
            else:
                t = improper_cell
                candidates = [
                    [i for i in range(order) if cube[i][t[1]][t[2]] == 1],
                    [j for j in range(order) if cube[t[0]][j][t[2]] == 1],
                    [k for k in range(order) if cube[t[0]][t[1]][k] == 1]
                ]
                c[0] = rng.choice(candidates[0])
                c[1] = rng.choice(candidates[1])
                c[2] = rng.choice(candidates[2])
            
            # Perform swaps
            cube[t[0]][t[1]][t[2]] += 1
            cube[t[0]][c[1]][c[2]] += 1
            cube[c[0]][c[1]][t[2]] += 1
            cube[c[0]][t[1]][c[2]] += 1
            cube[t[0]][t[1]][c[2]] -= 1
            cube[t[0]][c[1]][t[2]] -= 1
            cube[c[0]][t[1]][t[2]] -= 1
            cube[c[0]][c[1]][c[2]] -= 1
            
            is_proper = cube[c[0]][c[1]][c[2]] != -1
            if not is_proper:
                improper_cell = list(c)
            
            step += 1
        
        # Project cube to Latin square
        square = [[0 for _ in range(order)] for _ in range(order)]
        for x in range(order):
            for y in range(order):
                for s in range(order):
                    if cube[x][y][s] == 1:
                        square[x][y] = s
                        break

        return square

        # return [[self._items[value] for value in row] for row in square]


class CyclicSemigroup(Magma):
    def __init__(self, order, period, identity=False):
        def operation(a, b):
            c = a + b + (0 if identity else 1)
            while c >= order:
                c -= period
            return c
        table = [[operation(a, b) for b in range(order)] for a in range(order)]
        super().__init__(order, table=table)

class CyclicMonoid(CyclicSemigroup):
    def __init__(self, order, period):
        super().__init__(order, period, identity=True)

class AdjoinedSemigroup(Magma):
    '''
    Idea from Kulosman and Miller 2011 "Adjoining Idempotents to Semigroups".
    You can always create a semigroup by adding one more element "e" to an
    underlying semigroup. There are four simple rules that can be used for e:

    1. Add a zero, so that ae = ea = e for all a.
    2. Add an identity, so that ae = ea = a for all a.
    3. Imitate an existing element b, so that ae = ab, ea = ba for all a.
    4. Add an idempotent that imitates an idempotent b, like 3, but ee = e.
    '''
    def __init__(self, underlying, imitate, idempotent=False):
        elems = list(underlying.generate())
        indfor = {e: i for i, e in enumerate(elems)}
        newind = len(elems)
        if isinstance(imitate, int):
            imitate = elems[imitate]
        assert imitate in indfor or imitate in ['zero', 'identity']
        if idempotent and not isinstance(imitate, str):
            assert imitate * imitate == imitate
        def operation(a, b):
            if a == 'zero' or b == 'zero' or (a == 'identity' == b):
                return newind
            if a == 'identity':
                return indfor[b]
            if b == 'identity':
                return indfor[a]
            return indfor[a * b]
        table = []
        for a in elems + [imitate]:
            row = []
            for b in elems + [imitate]:
                row.append(operation(a, b))
            table.append(row)
        if idempotent:
            table[-1][-1] = newind
        super().__init__(underlying.order() + 1, table=table)

# Unit Test
if __name__ == '__main__':
    qg = Quasigroup(5)
    print("Quasigroup Operation Table:")
    print(qg.operation_table_str())
    assert qg.is_quasigroup()
    a = qg.generate()[0]
    b = qg.generate()[1]
    assert (a * (a // b)) == b
    assert ((a / b) * b) == a

    cs = CyclicSemigroup(6, 3)
    print("Cyclic Semigroup Operation Table:")
    print(cs.operation_table_str())
    assert cs.is_semigroup()
    assert not cs.is_monoid()

    cm = CyclicMonoid(8, 7)
    print("Cyclic Monoid (order 8, period 7) Operation Table:")
    print(cm.operation_table_str())
    assert cm.is_semigroup()
    assert cm.is_monoid()

    cs2 = CyclicSemigroup(8, 7)
    print("Cyclic Semigroup (order 8, period 7) Operation Table:")
    print(cs2.operation_table_str())
    assert cs2.is_semigroup()
    assert not cs2.is_monoid()

    from sympy.combinatorics import CyclicGroup, DihedralGroup, AlternatingGroup
    as0 = AdjoinedSemigroup(CyclicGroup(7), 'zero')
    print("Adjoined Semigroup (adding zero to C7) Operation Table:")
    print(as0.operation_table_str())
    assert as0.is_semigroup(True)
    assert as0.is_monoid()

    as1 = AdjoinedSemigroup(CyclicGroup(7), 1)
    print("Adjoined Semigroup (imitating generator in C7) Operation Table:")
    print(as1.operation_table_str())
    assert as1.is_semigroup(True)
    assert not as1.is_monoid()

    as2 = AdjoinedSemigroup(CyclicGroup(7), 0)
    print("Adjoined Semigroup (imitating identity C7) Operation Table:")
    print(as2.operation_table_str())
    assert as2.is_semigroup(True)
    assert not as2.is_monoid()

    as3 = AdjoinedSemigroup(CyclicGroup(7), 0, idempotent=True)
    print("Adjoined Semigroup (adjoining idempotent in C7) Operation Table:")
    print(as3.operation_table_str())
    assert as3.is_semigroup(True)
    assert as3.is_monoid()
