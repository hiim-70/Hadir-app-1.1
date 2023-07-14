import math

if __name__ == "__main__":
    NumOfEmp = int(input("enter the number of employees: "))
    lst = []
    for n in range(0,NumOfEmp):
        x = int(input("enter the hours of employee [{}] : ".format(n+1)))
        lst.append(x)

    for i,n in enumerate(lst):
        res = math.log(n) * (56/12)
        print('employee {} bonus : {}'.format(i+1,round(res,1)))

    print('Thank you')