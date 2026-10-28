"""Explicit finite-field HH instantiation for bounded baseline diagnostics.

FFDHE3072 prime from RFC 7919 Appendix A.2. The RFC estimates 125-bit group
strength; this is not a claim about an entire baseline's protocol security.
Generator points are hashed to the order-q subgroup without publishing logs.
"""
from __future__ import annotations
import hashlib
import gmpy2

FFDHE3072_P = int("""
FFFFFFFF FFFFFFFF ADF85458 A2BB4A9A AFDC5620 273D3CF1
D8B9C583 CE2D3695 A9E13641 146433FB CC939DCE 249B3EF9
7D2FE363 630C75D8 F681B202 AEC4617A D3DF1ED5 D5FD6561
2433F51F 5F066ED0 85636555 3DED1AF3 B557135E 7F57C935
984F0C70 E0E68B77 E2A689DA F3EFE872 1DF158A1 36ADE735
30ACCA4F 483A797A BC0AB182 B324FB61 D108A94B B2C8E3FB
B96ADAB7 60D7F468 1D4F42A3 DE394DF4 AE56EDE7 6372BB19
0B07A7C8 EE0A6D70 9E02FCE1 CDF7E2EC C03404CD 28342F61
9172FE9C E98583FF 8E4F1232 EEF28183 C3FE3B1B 4C6FAD73
3BB5FCBC 2EC22005 C58EF183 7D1683B2 C6F34A26 C1B2EFFA
886B4238 611FCFDC DE355B3B 6519035B BC34F4DE F99C0238
61B46FC9 D6E6C907 7AD91D26 91F7F7EE 598CB0FA C186D91C
AEFE1309 85139270 B4130C93 BC437944 F4FD4452 E2D74DD3
64F2E21E 71F54BFF 5CAE82AB 9C9DF69E E86D2BC5 22363A0D
ABC52197 9B0DEADA 1DBF9A42 D5C4484E 0ABCD06B FA53DDEF
3C1B20EE 3FD59D7C 25E41D2B 66C62E37 FFFFFFFF FFFFFFFF
""".replace(" ", "").replace("\n", ""), 16)
FFDHE3072_Q = (FFDHE3072_P-1)//2


class FiniteFieldHH:
    def __init__(self, dimension):
        if dimension < 1:
            raise ValueError("Positive dimension required")
        self.dimension, self.p, self.q = dimension, FFDHE3072_P, FFDHE3072_Q
        self.generators = []
        for index in range(dimension):
            counter = 0
            while True:
                material = b"point1-SA-DIAGNOSTIC-HH-v1|"+index.to_bytes(8, "big")+counter.to_bytes(4, "big")
                candidate = int.from_bytes(hashlib.shake_256(material).digest(392), "big") % self.p
                generator = candidate*candidate % self.p
                if generator not in (0, 1):
                    self.generators.append(generator)
                    break
                counter += 1

    def hash(self, vector):
        values = list(vector)
        if len(values) != self.dimension:
            raise ValueError("Wrong HH vector dimension")
        total = gmpy2.mpz(1)
        for generator, value in zip(self.generators, values):
            total = total*gmpy2.powmod(generator, int(value), self.p) % self.p
        return int(total)
