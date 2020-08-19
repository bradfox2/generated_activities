Predict unique but dependent categorical fields arranged as groups, issued in an sequence of arbitrary length.  Sequence and all elements therein is dependent on some static information provided once at start of sequence, similar to an image captioning problem.

 Example data is 4 batches of bptt 2, minibatch size 2,  3 categories
``` python
 data = [
     [
         [["sos", "sos", "sos"], ["sos", "sos", "sos"]],
         [["tx0", "stx0", "lx0"], ["ty0", "sty0", "ly0"]],
     ],
     [  # bptt record 1
         [["tx0", "stx0", "lx0"], ["ty0", "sty0", "ly0"]],
         [["tx1", "stx1", "lx1"], ["ty1", "sty1", "ly1"]],
     ],
     [  # bptt record 2
         [["tx1", "stx1", "lx1"], ["ty1", "sty1", "ly1"]],
         [["eos", "eos", "eos"], ["ty2", "sty2", "ly2"]],
     ],
     [  # bptt record N
         [["pad", "pad", "pad"], ["ty2", "sty2", "ly2"]],
         [["pad", "pad", "pad"], ["eos", "eos", "eos"]],
     ],
 ]
```