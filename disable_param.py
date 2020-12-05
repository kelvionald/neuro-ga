from ga import *

x_ignore_stat = []
y_ignore_stat = []

len_population = 10
for ignore_gen in range(0, 4):
    ignore_mutation_gens = [ignore_gen]
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
    y_ignore_stat.append(s)
    x_ignore_stat.append(ignore_gen)
    print('xMut, yMut', x_ignore_stat, y_ignore_stat)
    # break

plt.plot(x_ignore_stat, y_ignore_stat)
plt.show()
