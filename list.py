n=int(input())
for i in range(n):
    for j in range(n-i):
        print(" ",end="")
    p=1
    for j in range(i):
        print(p,end="")
        p=p+1
    p=i-1
    for j in range(i-1):
        print(p,end="")
        p=p-1
    print()
y=5
if(y==4&y==5):
    print('yes ')
else:
    print('no')
for i in range(10):
    print(i)