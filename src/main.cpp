#include <iostream>
#include <deque>
#include <vector>
#include <unordered_set>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <stack>
#include <queue>
#include <array>
#include <cstddef>
#include <limits>

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

        void removeDuplicates(Node* node)
        {
            std::unordered_set<T> uniqueValues;
            for (Node* prev = nullptr, *curr = node; curr; curr = curr->next)
            {
                if (uniqueValues.contains(curr->data))
                {
                    prev->next = curr->next;
                }
                else
                {
                    uniqueValues.emplace(curr->data);
                    prev = curr;
                }
            }
        }
        
        Node* nthFromEnd(Node* node, int n)
        {
            Node* first = node;
            Node* second = node;
            for (size_t i = 0; second && i < n; ++i)
            {
                second = second->next;
            }
            if (!second)
            {
                return nullptr;
            }
            while (second)
            {
                first = first->next;
                second = second->next;
            }
            return first;
        }

        Node* m_head = nullptr;
        Node* m_curr = nullptr;
    };

    void test()
    {
        SingleLinkedList<int> list;
        list.insert(1);
        list.insert(2);
        list.insert(4);
        list.insert(3);
        list.insert(4);
        list.insert(1);
        list.insert(5);
        list.insert(1);

        list.removeDuplicates(list.m_head);
        list.traverse(list.m_head);
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
        
        int maxDepth(Node* root)
        {
            if (!root)
                return 0;

            return 1 + std::max(maxDepth(root->left), maxDepth(root->right));
        }
        
        int minDepth(Node* root)
        {
            if (!root)
                return 0;

            return 1 + std::min(maxDepth(root->left), maxDepth(root->right));
        }
        
        bool isBalanced(Node* root)
        {
            return (maxDepth(root) - minDepth(root) <= 1);
        }
        
        std::vector<std::vector<Node*>> getLevels(Node* root)
        {
            std::vector<std::vector<Node*>> levels;
            levels.emplace_back({root});
            while (true)
            {
                std::vector<Node*>& prevLevel{levels.back()};
                std::vector<Node*> level;
                for (Node* n : prevLevel)
                {
                    if (n->left)
                        level.emplace_back(n->left);
                    if (n->right)
                        level.emplace_back(n->right);
                }
                if (level.empty())
                    break;
                levels.emplace_back(std::move(level));
            }
            return levels;
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

    // Horspul algorithm
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

namespace graphs
{
    using Graph = std::vector<std::vector<int>>;
    constexpr int Infinity = std::numeric_limits<int>::max();

    void dfs(const Graph& graph, int vertex)
    {
        std::unordered_set<int> used;
        used.emplace(vertex);

        std::vector<int> p(graph.size(), 0);
        std::vector<int> d(graph.size(), 0);

        std::deque<int> q;
        q.push_back(vertex);

        while (!q.empty())
        {
            int v = q.front();
            q.pop_front();

            std::cout << "visiting node=" << v << std::endl;

            for (int neighbour : graph[v])
            {
                if (!used.contains(neighbour))
                {
                    q.push_back(neighbour);
                    used.emplace(neighbour);

                    p[neighbour] = v;
                    d[neighbour] = d[v] + 1;
                }
            }
        }
    }

    void do_bfs_recursively(const Graph& graph, int vertex, std::unordered_set<int>& used, std::deque<int>& order)
    {
        std::cout << "visiting node=" << vertex << std::endl;
        used.emplace(vertex);

        for (int to : graph[vertex])
        {
            if (!used.contains(to))
                do_bfs_recursively(graph, to, used, order);
        }

        // for topological sort
        order.push_back(vertex);
    }

    void bfs_recursive(const Graph& graph)
    {
        std::deque<int> order;
        std::unordered_set<int> used;
        for (int to = 0; to < graph.size(); ++to)
        {
            do_bfs_recursively(graph, to, used, order);
        }
    }

    void bfs(const Graph& graph, int vertex)
    {
        std::unordered_set<int> used;
        std::stack<int> s;

        used.emplace(vertex);
        s.push(vertex);

        while (!s.empty())
        {
            int v = s.top();
            s.pop();

            std::cout << "visiting node=" << v << std::endl;
            used.emplace(v);

            for (int neighbour : graph[v])
            {
                if (!used.contains(neighbour))
                    s.push(neighbour);
            }
        }
    }

    // https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-using-priority_queue-stl/
    void dijkstra(const Graph& graph, int source)
    {
        using VertexInfo = std::pair<int, int>;
        std::priority_queue<VertexInfo, std::vector<VertexInfo>, std::greater<VertexInfo>> pq;
        std::vector<int> dist(graph.size(), Infinity);

        pq.push({0, source});
        dist[source] = 0;

        while (!pq.empty())
        {
            int u = pq.top().second;
            pq.pop();

            for (size_t i = 0; i < graph[u].size(); ++i)
            {
                if (graph[u][i] == Infinity)
                    continue;

                int v = i;
                int weight = graph[u][v];
                if (dist[v] > dist[u] + weight)
                {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
    }
    
    bool pathExists(const Graph& graph, int src, int dst)
    {
        std::unordered_set<int> visited;
        std::queue<int> q;

        visited.emplace(src);
        q.push(src);

        while (!q.empty())
        {
            int v = q.front();
            q.pop();

            for (const int neighbour : graph[v])
            {   
                if (dst == neighbour)
                    return true;

                if (!visited.contains(neighbour))
                {
                    q.push(neighbour);
                    visited.emplace(neighbour);
                }
            }
        }

        return false;
    }
}

namespace tests
{
    bool isUniqueChars(const std::string& str)
    {
        u_int64_t checker = 0;
        for (size_t i = 0; i < str.size(); ++i)
        {
            u_int8_t val = str[i] - 'a';
            if (checker & (1 << val))
                return false;
            checker |= (1 << val);
        }
        return true;
    }

    void reverse(char* str)
    {
        char* end = str;
        while (*end)
        {
            ++end;
        }
        --end;
        while (str < end)
        {
            char* tmp = str;
            *str++ = *end;
            *end-- = *tmp;
        }
    }

    void removeDuplicates(char* str)
    {
        char* begin = str;
        size_t tail = 0;
        for (u_int64_t checker = 0; *str; ++str)
        {
            const int val = *str - 'a';
            if ((checker & (1 << val)) == 0)
            {
                *(begin + tail++) = *str;
                checker |= (1 << val);
            }
        }
        *(begin + tail) = '\0';
    }
    
    // first second third 4|
    // first second third 4      |
    // first%20second%20third%204|
    void urlifySpaces(std::string& str)
    {
        if (str.empty())
            return;

        int spaces = 0;
        for (size_t i = 0; i < str.size(); ++i)
        {
            if (std::isspace(str[i]))
                ++spaces;
        }
        if (spaces == 0)
            return;
        
        int oldSize = str.size();
        str.resize(oldSize + spaces*2);
        for (int j = str.size() - 1, i = oldSize - 1; i >= 0 && j >= 0; --i)
        {
            if (std::isspace(str[i]))
            {
                str[j--] = '0';
                str[j--] = '2';
                str[j--] = '%';
            }
            else
            {
                str[j--] = str[i];
            }
        }
    }

    void test()
    {
        std::string str = "first second third 4";
        urlifySpaces(str);
        std::cout << str << std::endl;
    }
}

namespace cpp
{
    class A
    {
    public:
        A()
        {
            std::cout << "A::A();" << std::endl;
        }
        virtual ~A()
        {
            std::cout << "A::~A();" << std::endl;
        }

        A(A&&) = delete;
        A& operator=(A&&) = delete;

        A& operator=(const A& obj)
        {
            std::cout << "A::operator=;" << std::endl;
            if (&obj != this)
            {
               a = obj.a;
               if (b_size > 0)
                   delete [] b;
               b_size = obj.b_size;
               int* p = new int[b_size];
               std::memcpy(p, obj.b, sizeof(int)*b_size);
               b = p;
            }
            return *this;
        }

        A(const A& obj)
        {
            std::cout << "A::A(const A&);" << std::endl;
            (void)obj;
        }

        virtual void foo1() const
        {
            std::cout << "A::foo1;" << std::endl;
        }

        virtual void foo2() const
        {
            std::cout << "A::foo2;" << std::endl;
        }

    protected:
        int a = 0;
        int* b = nullptr;
        size_t b_size = 0;
    };

    class B : public A
    {
    public:
        B()
        {
            std::cout << "B::B();" << std::endl;
        }
        B(int _a, int* _b, int _bSize)
        {
            A::a = _a;
            A::b = _b;
            A::b_size = _bSize;
            std::cout << "B::B(int, int*, int);" << std::endl;
        }
        B(const B&)
        {
            std::cout << "B::B(const B&);" << std::endl;
        }

        ~B() override
        {
            std::cout << "B::~B;" << std::endl;
        }

        B(B&&) = delete;
        B& operator=(B&&) = delete;

        void foo1() const override
        {
            std::cout << "B::foo1;" << std::endl;
        }

        void foo2(int) const
        {
            std::cout << "B::foo2(int);" << std::endl;
        }

    private:
        bool a = false;
        std::string b;
    };

    void params1(A a)
    {
        std::cout << "params1" << std::endl;
        a.foo1();
    }

    void params2(const A& a)
    {
        std::cout << "params2" << std::endl;
        a.foo1();
    }

    void polymorphysm1(A* a)
    {
        a->foo1();
        a->foo2();
    }

    void polymorphysm2(A a)
    {
        a.foo1();
        a.foo2();
    }

    void polymorphysm3(B b)
    {
        b.foo1();
        b.A::foo2();
    }

    void polymorphysm4(B* b)
    {
        b->foo1();
        b->foo2(0);
    }

    void test()
    {
        A a1;
        params1(a1);
        params2(a1);
        A a2 = a1;
        A a3(a2);
        A a4{a3};
        params1(a4);

        B b1;
        a4 = b1;
        B b2{0, nullptr, 0};
        B b3{b2};
        params1(b1);
        params2(b3);
    }

    void test2()
    {
        A* a = new A;
        polymorphysm1(a);
        polymorphysm2(*a);

        A* b = new B;
        polymorphysm1(b);
        polymorphysm2(*b);

        B* b2 = new B;
        polymorphysm1(b2);
        polymorphysm2(*b2);
        polymorphysm3(*b2);
        polymorphysm4(b2);
    }

    A copymove1()
    {
        A a;
        a.foo1();
        return a;
    }

    void test3()
    {
        A a = copymove1();
        a.foo1();
    }

    void copy_and_throw(A a)
    {
        std::cout << "copy_and_throw()" << std::endl;
        throw std::runtime_error("too bad..");
    }

    void test4()
    {
        A a;
        try
        {
            copy_and_throw(a);
        }
        catch (const std::exception& e)
        {
            std::cout << e.what() << std::endl;
        }
    }
}

int main()
{
    cpp::test4();
    return 0;

    graphs::Graph graph =
    {
        {1, 2, 3},
        {1, 3, 5},
        {6, 7},
        {7, 6},
        {0, 1},
        {3, 1, 2},
        {2, 3, 4},
        {4},
    };

    graphs::bfs_recursive(graph);
    graphs::bfs(graph, 0);
    return 0;
}
