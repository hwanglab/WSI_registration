import time
start = time.time()

for y in range(1379):
    for x in range(2444):
        if x>=0 and x<2444 and y>=0 and y<1379:
            pass
        
end = time.time()
print(end-start)
temp= end-start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('%d:%d:%d'%(hours,minutes,seconds))

