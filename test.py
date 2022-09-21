from torchquad import MonteCarlo, set_up_backend, Trapezoid
set_up_backend("torch", data_type="float32")
mc = MonteCarlo()
tp = Trapezoid()

print(tp.integrate(lambda x:x**2+1,1,1000,[[0,1]]))
