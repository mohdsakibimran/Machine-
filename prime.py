def Solve(k, arr):
    # Write code here
    n = len(arr)
    j = 0
    for i in range(n):
        if arr[i] < 0:
            continue
        while arr[i] != 0 and i + k >= j:
            if arr[j] >= 0:
                j += 1
            else:
                x = min(arr[i], -1 * arr[j])
                arr[i] -= x
                arr[j] += x

    ans = 0
    for i in range(n):
        ans += abs(arr[i])

    return ans



n, k = [int(x) for x in input().split()]
arr = [int(x) for x in input().split()]

out_ = Solve(k, arr)
print(out_)