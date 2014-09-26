epsilon = .1
opponent_price = .0951

# P[0] is the probability that there is a sale with a display minus the probability that there is a sale without a display when there hasn't been any display yet.
# P[1] is the probability that there is a sale with a display minus the probability that there is a sale without a display when there has been one display so far.
P = [.5, .5]
# Q[0] is the probability that there is a sale with a display minus the probability that there is a sale without a display for the first display.
# Q[1] is the probability that there is a sale with a display minus the probability that there is a sale without a display when there hasn't been a display at the first opportunity.
# Q[2] is the probability that there is a sale with a display minus the probability that there is a sale without a display when there has been a display already.
Q = [1., 1., 1.]

# What is the probability of each of the 4 following events: 00, 01, 10, 11
def compute_fP(P, epsilon, opponent_price):
    frequencies = [0., 0., 0., 0.]
    # Probability that you do not buy the display given that there has been no display yet.
    P_0_0 = float(P[0] < opponent_price) * (1 - epsilon) + epsilon/2
    # Probability that you buy the display given that there has been no display yet.
    P_0_1 = 1 - P_0_0
    # Probability that you do not buy the display given that there has been a display.
    P_1_0 = float(P[1] < opponent_price) * (1 - epsilon) + epsilon/2
    # Probability that you do buy the display given that there has been a display.
    P_1_1 = 1 - P_1_0

    frequencies[0] = P_0_0**2
    frequencies[1] = P_0_0 * P_0_1
    frequencies[2] = P_0_1 * P_1_0
    frequencies[3] = P_0_1 * P_1_1

    return frequencies

# What is the probability of each of the 4 following events: 00, 01, 10, 11
def compute_fQ(Q, epsilon, opponent_price):
    frequencies = [0., 0., 0., 0.]
    # Probability that you do not buy the first display.
    Q_0 = float(Q[0] < opponent_price) * (1 - epsilon) + epsilon/2
    # Probability that you buy the first display.
    Q_1 = 1 - Q_0
    # Probability that you do not buy the second display given that you did not buy the first one.
    Q_0_0 = float(Q[1] < opponent_price) * (1 - epsilon) + epsilon/2
    # Probability that you do buy the second display given that you did not buy the first one.
    Q_0_1 = 1 - Q_0_0
    # Probability that you do not buy the second display given that you did buy the first one.
    Q_1_0 = float(Q[2] < opponent_price) * (1 - epsilon) + epsilon/2
    # Probability that you do buy the second display given that you did buy the first one.
    Q_1_1 = 1 - Q_1_0

    frequencies[0] = Q_0 * Q_0_0
    frequencies[1] = Q_0 * Q_0_1
    frequencies[2] = Q_1 * Q_1_0
    frequencies[3] = Q_1 * Q_1_1

    return frequencies


for i in range(4):
    fP = compute_fP(P, epsilon, opponent_price)
    # P[0] is the probability that there is a sale with a display minus the probability that there is a sale without a display when there hasn't been any display yet.
    P[0] = 1. - fP[1]/(fP[1] + 2*fP[0])
    # P[1] is the probability that there is a sale with a display minus the probability that there is a sale without a display when there has been one display so far.
    P[1] = 0

    fQ = compute_fQ(Q, epsilon, opponent_price)
    # Q[0] is the probability that there is a sale with a display minus the probability that there is a sale without a display for the first display.
    Q[0] = 1. - fQ[1]/(fQ[0] + fQ[1])
    # Q[1] is the probability that there is a sale with a display minus the probability that there is a sale without a display when there hasn't been a display at the first opportunity.
    Q[1] = 1.
    # Q[2] is the probability that there is a sale with a display minus the probability that there is a sale without a display when there has been a display already.
    Q[2] = 0.

    print("fP = {}, P = {}".format(fP, P))
    print("fQ = {}, Q = {}".format(fQ, Q))