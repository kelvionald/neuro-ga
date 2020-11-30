from ga import *

xPop = []
yPop = []
kMutation = 30
for pop in range(2, 25, 2):
    xPop.append(pop)
    len_population = pop
    history_col_mutants = []
    history_obj_func = []
    history_epochs = []

    new = []
    old = []
    t = init_population()
    print('---- INITIALIZED ----')

    best = [[], 0]
    for epoch in range(1, iter_epochs):
        new = t
        t = operator_roulette(t)
        t = operator_crossingover(t)
        t = operator_mutation(t)
        old = t + new
        t = operator_selection(old)
        history_epochs.append(epoch)

        tt = list(map(lambda x: [x, objective_function(x)], t))
        tt = sorted(tt, key=lambda x: -x[1])
        el, acc = tt[0]
        if best[1] < acc:
            best = [el, acc]
        print(f'Epoch {epoch}')
        prtElement('CURR BEST', el, [0, acc])
        prtElement('ALL  BEST', best[0], [0, best[1]])
        print('--------')

    s = 0
    for et in tt:
        s += et[1]
    s /= len(tt)
    yPop.append(s)
    print('xPop, yPop', xPop, yPop)

plt.plot(xPop, yPop)
plt.show()
