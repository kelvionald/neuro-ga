from ga import *

len_population = 10

iter_epochs = 100


new = []
old = []
t = init_population()
print('---- INITIALIZED ----')

global best
best = [[], 0]

def prtEl(tt):
    global best
    el, acc = tt
    if best[1] < acc:
        best = [el, acc]
    prtElement('CURR BEST', el, [0, acc])

for epoch in range(1, iter_epochs):
    with open('epoch.txt', 'w') as f:
        f.write(str(epoch))
    new = t
    t = operator_roulette(t)
    t = operator_crossingover(t)
    t = operator_mutation(t)
    old = t + new
    t = operator_selection(old)
    history_epochs.append(epoch)

    tt = list(map(lambda x: [x, objective_function(x)], t))
    tt = sorted(tt, key=lambda x: -x[1])

    print(f'Epoch {epoch}')
    prtEl(tt[0])
    prtEl(tt[1])
    prtEl(tt[2])
    prtElement('ALL  BEST', best[0], [0, best[1]])

plt.scatter(history_epochs, history_col_mutants)
plt.show()
plt.plot(history_epochs, history_obj_func)
plt.show()