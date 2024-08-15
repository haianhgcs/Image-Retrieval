# Image-Retrieval
## Giới thiệu
Truy vấn hình ảnh (Images Retrieval) là một bài toán thuộc lĩnh vực Truy vấn thông tin (Information Retrieval). Trong đó, nhiệm vụ của ta là xây dựng một chương trình trả về các hình ảnh (Images) có liên quan đến hình ảnh truy vấn đầu vào(Query) và các hình ảnh được lấy từ một bộ dữ liệu hình ảnh cho trước, hiện nay có một số ứng dụng truy vấn ảnh như: Google Search Image, chức năng tìm kiếm sản phẩm bằng hình ảnh trên Shopee, Lazada, Tiki, ...

![alt text](./readme/ImageRetrievalUsecase.PNG)

## Pipeline
Có rất nhiều cách thiết kế hệ thống truy vấn hình ảnh khác nhau, tuy nhiên về mặt tổng quát sẽ có pipeline như sau:
![alt text](./readme/ImageRetrievalPipeline.PNG)

Input/Output của một hệ thống truy vấn hình ảnh bao gồm:
* Input: Hình ảnh truy vấn Query Image và bộ dữ liệu Images Library.
* Output: Danh sách các hình ảnh có sự tương tự đến hình ảnh truy vấn.

Trong dự án này, chúng ta sẽ xây dựng một hệ thống truy xuất hình ảnh bằng cách sử dụng mô hình deep learning đã được huấn luyện trước (CLIP) để trích xuất đặc trưng của ảnh và thu được các vector đặc trưng. Sau đó, chúng ta sẽ sử dụng vector database (Chroma) để index, lưu trữ và truy xuất các ảnh tương tự với ảnh yêu cầu thông qua các thuật toán đo độ tương đồng.

![alt text](./readme/ImageRetrievalPipeline_project.PNG)

## Mục tiêu của dự án
Các mục tiêu chính của dự án bao gồm:
* Xây dựng chương trình truy vấn ảnh cơ bản.
* Phát triển chương trình truy vấn ảnh nâng cao với CLIP model và vector database.
* (Optional) Thu thập và xử lý dữ liệu nhằm mục đích xây dựng chương trình truy vấn ảnh cá nhân hóa.