Predict unique but dependent categorical fields arranged as groups, issued in an sequence of arbitrary length

# Example data is 4 batches of bptt 2, minibatch size 2,  3 categories
# data = [
#     [
#         [["sos", "sos", "sos"], ["sos", "sos", "sos"]],
#         [["tx0", "stx0", "lx0"], ["ty0", "sty0", "ly0"]],
#     ],
#     [  # bptt record 1
#         [["tx0", "stx0", "lx0"], ["ty0", "sty0", "ly0"]],
#         [["tx1", "stx1", "lx1"], ["ty1", "sty1", "ly1"]],
#     ],
#     [  # bptt record 2
#         [["tx1", "stx1", "lx1"], ["ty1", "sty1", "ly1"]],
#         [["eos", "eos", "eos"], ["ty2", "sty2", "ly2"]],
#     ],
#     [  # bptt record N
#         [["eos", "eos", "eos"], ["ty2", "sty2", "ly2"]],
#         [["eos", "eos", "eos"], ["eos", "eos", "eos"]],
#     ],
# ]