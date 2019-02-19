from playground.dqn.experience_buffer import Experience_Buffer

BUFFER_SIZE = 5
SAMPLE_TIMES = 10
eb = Experience_Buffer(BUFFER_SIZE)

# Assume exp is made up of an array of length BUFFER_SIZE
for i in range(BUFFER_SIZE):
    eb.add([i,i+1,i+2,i+3,0.01 * i])

sample_exp = eb.sample(5)
expected_val = [[0, 1, 2, 3, 0.0], 
                [1, 2, 3, 4, 0.01], 
                [2, 3, 4, 5, 0.02], 
                [3, 4, 5, 6, 0.03], 
                [4, 5, 6, 7, 0.04]]
assert (sample_exp == expected_val)

sample_exp = eb.sample(0)
assert(sample_exp == [])



