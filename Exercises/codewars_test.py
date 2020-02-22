def in_array(array1, array2):
    mlist = []
    for i in array1:
        for j in array2:
            print(i)
            print(j)
            print("-----")
            if i in j:
                if i == j: 
                    continue
                mlist.append(i) 
                break
    return sorted(mlist)

a1 = ["live", "arp", "strong"] 
a2 = ["lively", "alive", "harp", "sharp", "armstrong"]

a = 12

def expanded_form(num):
    result = []
    divider = 10
    while divider < num:
        temp = num%divider
        print(temp)
        if temp != 0:
            result.insert(0, str(temp))
        num -= temp
        divider *= 10
    result.insert(0, str(num))
    return '+'.join(result)

def order_weight(strng):
    sum_list = []
    num_sum = 0
    weight_list = strng.split(" ")
    for i in weight_list:
        int_i = int(i)
        while(int_i > 0):
            remainder = int_i % 10
            num_sum = remainder + num_sum
            int_i = int_i // 10 
        sum_list.append(num_sum)

    sum_list.sort()
    return sum_list




print(order_weight("103 123 4444 99 2000"))

#print(expanded_form(a))
#print(in_array(a1, a2))

