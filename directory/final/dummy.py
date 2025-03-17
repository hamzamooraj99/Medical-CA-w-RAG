uq = "Why am I having trouble breathing?"
ir = "The image shows a lung suffering from pneumonia."

periods = ['.', '?', '!']

if(uq[-1] not in periods):
    uq = uq + "."
else:
    pass

print(uq)