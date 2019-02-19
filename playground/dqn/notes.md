Make sure the next_state is properly set to prevent the Q-values from being all the same.

Pyplot drawing graphs without blocking the main thread

```
plt.axis([-50,50,0,10000])
    plt.ion()
    plt.show()

    x = np.arange(-50, 51)
    for pow in range(1,5):   # plot x^1, x^2, ..., x^4
        y = [Xi**pow for Xi in x]
        plt.plot(x, y)
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
```

need to normalize the input data, because volume and price are not on the same scale, which makes training hard to converge to global min

never leave the epsilon the same across episodes, as it will stay at the min value and cause the agent to fully trust the actions output by the q-network