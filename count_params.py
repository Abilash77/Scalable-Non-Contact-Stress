import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from DL_models import get_model

input_shapes = {
    'audio': (10, 169),
    'face': (10, 12),
    'keystroke': (10, 7),
    'handwriting': (10, 9),
    'eye': (10, 5)
}

model = get_model('fusion', input_shapes=input_shapes, num_classes=2)

import io
with io.StringIO() as buf:
    model.summary(print_fn=lambda x: buf.write(x + '\n'))
    summary = buf.getvalue()

print("TOTAL PARAMS:", model.count_params())
print(summary)
