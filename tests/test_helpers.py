import numpy as np  

def grad_calc_activ(x, layer, eps=1e-8):
    h = x * np.sqrt(eps)
    xp = x + h
    xm = x - h

    p = layer.f(xp)
    m = layer.f(xm)

    approx = (p - m) / (2 * h)

    grad = layer.f_prime(x)

    num = np.linalg.norm(approx - grad)
    denum = np.linalg.norm(approx) + np.linalg.norm(grad)

    return num / denum

def cost(pred, target):
    cst = np.sum((pred - target) ** 2)
    return cst

def grad_calc_layers(x, y, net, eps=1e-8):
    tests = dict()
    for k in range(len(net.layers)):
        layer = net.layers[k]
        if layer.params != {}:
            for par in layer.params.keys():
                theta = layer.params[par]
                shp = theta.shape
                theta = theta.flatten()

                approx = []
                out = net.forward(x)[0]
                out = (out - y) * 2
                net.backward(out)[0]
                grad = net.layers[k].grads[par].flatten()

                for i in range(len(theta)):
                    thetaplus = np.copy(theta)
                    thetaplus[i] = thetaplus[i] + eps
                    thetaminus = np.copy(theta)
                    thetaminus[i] = thetaminus[i] - eps

                    net.layers[k].params[par] = thetaplus.reshape(shp)
                    o1 = net.forward(x)[0]
                    cost1 = cost(o1, y)

                    net.layers[k].params[par] = thetaminus.reshape(shp)
                    o2 = net.forward(x)[0]
                    cost2 = cost(o2, y)

                    approx.append((cost1 - cost2) / (2 * eps))
                    
                num = np.linalg.norm(approx - grad)
                denum = np.linalg.norm(approx) + np.linalg.norm(grad)

                g = num / denum

                tests[par] = g
    return tests
