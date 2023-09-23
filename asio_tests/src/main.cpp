#include <boost/asio.hpp>
#include <iostream>

using namespace std;
using namespace boost::asio;

void test()
{
    io_service service;
    ip::tcp::endpoint ep(ip::address::from_string("127.0.0.1"), 2023);
    ip::tcp::socket sock(service);
    sock.async_connect(ep, [](const boost::system::error_code& ec){
        std::cout << ec.message() << std::endl;
    });
    service.run();

    ip::tcp::socket tcp_socket(service);
    tcp_socket.open(ip::tcp::v4());
    tcp_socket.connect(ep);
    tcp_socket.write_some(boost::asio::buffer("Big chunk of data."));

    char data[1024];
    tcp_socket.available();
    tcp_socket.read_some(buffer(data, 1024));
    tcp_socket.shutdown(ip::tcp::socket::shutdown_receive);

    std::size_t bytes_left = tcp_socket.available();
    std::cout << bytes_left << std::endl;
    io_service::strand strand(service);
}

int main()
{
    test();
    return 0;
}
