#include <iostream>
#include <deque>
#include <vector>
#include <unordered_set>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace lists
{
    template <typename T>
    class DoubleLinkedList
    {
    public:
        struct Node
        {
            T data = {};
            Node* next = nullptr;
            Node* prev = nullptr;
        };

        void insert(const T& v)
        {
            Node* node = new Node{};
            node->data = v;
            node->prev = m_curr;
            if (!m_head)
            {
                m_head = node;
                m_curr = m_head;
            }
            else
            {
                m_curr->next = node;
                m_curr = node;
            }
        }

        bool remove(const T& v)
        {
            for (Node* node = m_head; node; node = node->next)
            {
                if (node->data == v)
                {
                    if (node->prev)
                        node->prev->next = node->next;
                    if (node->next)
                        node->next->prev = node->prev;
                    delete node;
                    return true;
                }
            }
            return false;
        }

    private:
        Node* m_head = nullptr;
        Node* m_curr = nullptr;
    };

    template <typename T>
    struct SingleLinkedList
    {
        struct Node
        {
            T data = {};
            Node* next = nullptr;
        };

        void insert(const T& v)
        {
            Node* node = new Node{};
            node->data = v;
            if (!m_head)
            {
                m_head = node;
                m_curr = m_head;
            }
            else
            {
                m_curr->next = node;
                m_curr = node;
            }
        }

        bool remove(const T& v)
        {
            Node* prev = nullptr;
            Node* curr = m_head;
            for (; curr; curr = curr->next)
            {
                if (curr->data == v)
                {
                    if (prev)
                        prev->next = curr->next;
                    else
                        m_head->next = curr->next;
                    delete curr;
                    return true;
                }
                prev = curr;
            }
            return false;
        }

        using Node = typename SingleLinkedList<T>::Node;

        Node* reverse(Node* head)
        {
            Node* newHead{nullptr};
            Node* curr{head};
            Node* prev{nullptr};
            while (curr)
            {
                Node* realNext{curr->next};
                newHead = curr;
                newHead->next = prev;
                prev = curr;
                curr = realNext;
            }
            return newHead;
        }

        void traverse(Node* head)
        {
            for (Node* n = head; n; n = n->next)
            {
                std::cout << n->data << std::endl;
            }
        }

        Node* m_head = nullptr;
        Node* m_curr = nullptr;
    };

    void test()
    {
        SingleLinkedList<int> list;
        list.insert(1);
        list.insert(2);
        list.insert(3);
        list.insert(4);
        list.insert(5);

        auto newList = list.reverse(list.m_head);
        list.traverse(newList);
    }
}

namespace trees
{
    template <typename T>
    class BinaryTree
    {
    public:
        struct Node
        {
            T data;
            Node* parent = nullptr;
            Node* left = nullptr;
            Node* right = nullptr;
        };

        Node* insert(const T& v)
        {
            Node* last = nullptr;
            for (Node* curr = m_root; curr;)
            {
                last = curr;
                if (curr->data == v)
                    return curr;
                if (v < curr->data)
                    curr = curr->left;
                else
                    curr = curr->right;
            }

            Node* node = new Node;
            node->data = v;
            node->parent = last;

            if (last)
            {
                if (v < last->data)
                    last->left = node;
                else
                    last->right = node;
            }
            else
                m_root = node;

            return node;
        }

        void traverse(Node* n)
        {
            if (!n)
                return;
            traverse(n->left);
            std::cout << n->data << std::endl;
            traverse(n->right);
        }

        void traverse_bfs()
        {
            std::deque<Node*> q;
            q.push_back(m_root);
            while (!q.empty())
            {
                Node* n = q.front();
                q.pop_front();
                if (n->left)
                    q.push_back(n->left);
                if (n->right)
                    q.push_back(n->right);
                std::cout << n->data << std::endl;
            }
        }

        Node* successor(Node* node)
        {
            if (node->right)
                return min(node->right);

            Node* p = node->parent;
            for (Node* n = node; p; p = p->parent)
            {
                if (p->left == n)
                    break;
                n = p;
            }
            return p;
        }

        Node* predecessor(Node* node)
        {
            if (node->left)
                return max(node->left);

            Node* p = node->parent;
            for (Node* n = node; p; p = p->parent)
            {
                if (p->right == n)
                    break;
                n = p;
            }
            return p;
        }

        Node* min(Node* node)
        {
            Node* min = nullptr;
            for (Node* n = node; n; n = n->left)
                min = n;
            return min;
        }

        Node* max(Node* node)
        {
            Node* max = nullptr;
            for (Node* n = node; n; n = n->right)
                max = n;
            return max;
        }

        Node* begin() const { return m_root; }

    private:
        Node* m_root = nullptr;
    };

    void test()
    {
        BinaryTree<int> tree;
        auto* node2 = tree.insert(2);
        auto* node1 = tree.insert(1);
        auto* node3 = tree.insert(3);
        auto* node5 = tree.insert(5);
        auto* node4 = tree.insert(4);
        auto* node0 = tree.insert(0);

        tree.traverse(tree.begin());
        tree.traverse_bfs();

        std::cout << tree.min(tree.begin())->data << std::endl;
        std::cout << tree.max(tree.begin())->data << std::endl;
        std::cout << std::endl;

        std::cout << tree.successor(node0)->data << std::endl;
        std::cout << tree.successor(node1)->data << std::endl;
        std::cout << tree.successor(node2)->data << std::endl;
        std::cout << tree.successor(node3)->data << std::endl;
        std::cout << tree.successor(node4)->data << std::endl;
        std::cout << tree.successor(node5) << std::endl;
        std::cout << std::endl;

        std::cout << tree.predecessor(node0) << std::endl;
        std::cout << tree.predecessor(node1)->data << std::endl;
        std::cout << tree.predecessor(node2)->data << std::endl;
        std::cout << tree.predecessor(node3)->data << std::endl;
        std::cout << tree.predecessor(node4)->data << std::endl;
        std::cout << tree.predecessor(node5)->data << std::endl;
        std::cout << std::endl;
    }
}

namespace heap
{
    template <typename T>
    class Heap
    {
    // https://habr.com/ru/articles/112222/
    public:
        Heap() = default;

        explicit Heap(const std::vector<T>& src)
        {
            buildFrom(src);
        }

        static void sort(std::vector<T>& src)
        {
            Heap heap{src};
            for (size_t i = src.size() - 1; i >= 0 ; --i)
            {
                heap.m_heap[i] = heap.popMax();
                heap.heapify(0);
            }
            src.swap(heap.m_heap);
        }

        void buildFrom(const std::vector<T>& src)
        {
            m_heap = src;
            for (size_t i = m_heap.size()/2; i >= 0; --i)
            {
                heapify(i);
            }
        }

        void add(const T& v)
        {
            m_heap.emplace_back(v);
            for (size_t i = m_heap.size() - 1; i > 0;)
            {
                size_t parentIndex = getParentIndex(i);
                if (m_heap[i] > m_heap[parentIndex])
                    std::swap(m_heap[i], m_heap[parentIndex]);
                else
                    break;
                i = parentIndex;
            }
        }

        void heapify(int i = 0)
        {
            while (i < m_heap.size())
            {
                size_t rightIndex{getRightIndex(i)};
                size_t leftIndex{getLeftIndex(i)};
                if (m_heap[i] > std::max(m_heap[rightIndex], m_heap[leftIndex]))
                    return;
                size_t biggerNodeIndex = m_heap[rightIndex] > m_heap[leftIndex] ? rightIndex : leftIndex;
                std::swap(m_heap[biggerNodeIndex], m_heap[i]);
                i = biggerNodeIndex;
            }
        }

        void output()
        {
            for (size_t i = 0; i < m_heap.size(); ++i)
            {
                std::cout << m_heap[i] << std::endl;
            }
            std::cout << std::endl;
        }

        size_t max() const { return m_heap.front(); }

    private:
        T popMax()
        {
            const T maxValue = max();
            m_heap.front() = m_heap.back();
            m_heap.pop_back();
            // Will be called in sort().
            // heapify(0);
            return maxValue;
        }
        static size_t getLeftIndex(size_t i) { return i*2+1; }
        static size_t getRightIndex(size_t i) { return i*2+2; }
        static size_t getParentIndex(size_t i) { return (i-1)/2; }

        std::vector<T> m_heap;
    };

    void test()
    {
        Heap<int> heap({1, 3, 2, 5, 7, 4, 9, 8});
        heap.add(0);
        heap.add(10);
        heap.add(13);
        heap.add(11);
        heap.add(12);
        heap.output();

        std::cout << heap.max() << std::endl;

        std::vector<int> arr{1, 3, 2, 5, 7, 4, 9, 8};
        Heap<int>::sort(arr);
        for (int item : arr)
            std::cout << item << std::endl;
    }
}

namespace common
{
    template<typename T>
    int find(const std::vector<T>& a, const T& v)
    {
        size_t left = 0;
        size_t right = a.size() - 1;
        while (left <= right)
        {
            size_t middle = (right - left) / 2 + left;
            if (v == a[middle])
                return middle;
            if (v > a[middle])
                left = middle + 1;
            else
                right = middle - 1;
        }
        return -1;
    }

    template <typename T>
    void rotate(std::vector<T>& a, size_t shift)
    {
        std::vector<T> tmp(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            tmp[i] = a[(i + shift) % a.size()];
        }
        a.swap(tmp);
    }

    std::vector<int> twoSum(std::vector<int>& a, int target)
    {
        std::sort(a.begin(), a.end());
        for (size_t l = 0, r = a.size() - 1; l < r;)
        {
            int sum = a[l] + a[r];
            if (sum == target)
                return {a[l], a[r]};
            if (sum < target)
                l++;
            else
                --r;
        }
        return {};
    }

    std::pair<size_t, int> findRange(std::vector<int>& a, int target)
    {
        if (a.empty())
            return {};
        int sum = 0;
        for (size_t l = 0, r = 0; l <= r && r < a.size();)
        {
            if (sum == target)
                return{l, r-l};
            if (sum < target)
                sum += a[r++];
            else
                sum -= a[l++];
        }
        return {};
    }

    void removeAll(std::vector<int>& items)
    {
        size_t i = 0;
        for (int item : items)
        {
            if (item)
                items[i++] = item;
        }
        while (i < items.size())
            items[i++] = 0;
    }

    void shiftZeroes(std::vector<int>& a)
    {
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (a[i] == 0)
            {
                size_t j = i;
                while (j < a.size() and a[j] == 0) ++j;
                if (j < a.size())
                    std::swap(a[i], a[j]);
            }
        }
    }

    void insertionSort(std::vector<int>& a)
    {
        for (size_t i = 1; i < a.size(); ++i)
        {
            for (size_t j = i; j >= 0 and a[j] < a[j-1]; --j)
                std::swap(a[j], a[j-1]);
        }
    }

    void bubbleSort(std::vector<int>& a)
    {
        for (size_t i = 0; i < a.size(); ++i)
        {
            for (size_t j = 1; j < a.size() - i; ++j)
                if (a[j - 1] > a[j])
                    std::swap(a[j], a[j - 1]);
        }
    }

    void shakerSort(std::vector<int>& a)
    {
        for (int l = 0, r = a.size() - 1; l < r; ++l, --r)
        {
            for (int i = l + 1; i <= r; ++i)
            {
                if (a[i - 1] > a[i])
                    std::swap(a[i], a[i - 1]);
            }
            for (int i = r - 1; i >= l; --i)
            {
                if (a[i + 1] < a[i])
                    std::swap(a[i + 1], a[i]);
            }
        }
    }

    void quickSort(std::vector<int>& a, int l, int r)
    {
        int i = l, j = r;
        int pivot = a[(l + r) / 2];
        while (i <= j)
        {
            while (a[i] < pivot) ++i;
            while (a[j] > pivot) --j;
            if (i <= j)
            {
                std::swap(a[i], a[j]);
                ++i;
                --j;
            }
        }
        if (j > l)
            quickSort(a, l, j);
        if (i < r)
            quickSort(a, i, r);
    }

    void mergeSort(int* a, int size)
    {
        if (size > 1)
        {
            int leftSize = size / 2;
            int rightSize = size - leftSize;
            mergeSort(a, leftSize);
            mergeSort(a + leftSize, rightSize);

            std::vector<int> out;
            out.reserve(size);

            int li = 0, ri = rightSize;
            while (li < leftSize && ri < size)
            {
                if (a[li] < a[ri])
                    out.emplace_back(a[li++]);
                else
                    out.emplace_back(a[ri++]);
            }
            if (li < leftSize)
                out.insert(out.end(), a + li, a + leftSize);
            if (ri < size)
                out.insert(out.end(), a + ri, a + size);
            
            for (int i = 0; i < size; ++i)
                a[i] = out[i];
        }
    }

    std::size_t substrPos(const std::string& str, const std::string& substr)
    {
        if (substr.empty())
            return std::string::npos;

        std::array<int, 256> offsets;
        std::fill(offsets.begin(), offsets.end(), substr.length());

        std::unordered_set<char> chars;
        for (auto ch : substr)
            chars.insert(ch);

        for (size_t i = 0; i < substr.size(); ++i)
            if (chars.find(substr[i]) != chars.end())
                offsets[substr[i]] = substr.size() - i - 1;

        const size_t lastPos{substr.size() - 1};
        for (size_t i = lastPos; i < str.size();)
        {
            size_t j = lastPos;
            while (j >= 0 && i >= 0 && str[i] == substr[j])
            {
                if (j == 0)
                    return i;
                --i;
                --j;
            }
            i += offsets[str[i]];
        }

        return std::string::npos;
    }
}

int main()
{
    size_t pos = common::substrPos("substring", "string");
    std::cout << pos << std::endl;

    pos = common::substrPos("abrac", "ra");
    std::cout << pos << std::endl;

    pos = common::substrPos("abbabgkdkdafkl", "dafk");
    std::cout << pos << std::endl;

    return 0;
}
