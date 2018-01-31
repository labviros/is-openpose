CXX = clang++
CXXFLAGS += -std=c++14 -Wall
LDFLAGS += -I/usr/local/include -L/usr/local/lib \
	-lprotobuf -pthread -lpthread -lrabbitmq -lSimpleAmqpClient \
	-lopenpose -lopencv_core -lopencv_imgcodecs \
	-lboost_system -lboost_chrono -lboost_program_options -lboost_filesystem \
	-lismsgs  -lopentracing -lzipkin -lzipkin_opentracing

PROTOC = protoc
LOCAL_PROTOS_PATH = ./msgs/

vpath %.proto $(LOCAL_PROTOS_PATH)

MAINTAINER = viros
SERVICE = skeleton
TEST = test
VERSION = 1
LOCAL_REGISTRY = ninja.local:5000

all: debug

debug: CXXFLAGS += -g 
debug: LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
debug: $(SERVICE) $(TEST)

release: CXXFLAGS += -Werror -O2
release: $(SERVICE)

$(SERVICE): skeleton.pb.o $(SERVICE).o
	$(CXX) $^ $(LDFLAGS) -o $@

$(TEST): skeleton.pb.o $(TEST).o
	$(CXX) $^ $(LDFLAGS) -o $@

.PRECIOUS: %.pb.cc
%.pb.cc: %.proto
	$(PROTOC) -I $(LOCAL_PROTOS_PATH) --cpp_out=. $<

clean:
	rm -f *.o *.pb.cc *.pb.h $(SERVICE) $(TEST)

docker: 
	docker build -t $(MAINTAINER)/$(SERVICE):$(VERSION) --build-arg=SERVICE=$(SERVICE) .

push_local: docker
	docker tag $(MAINTAINER)/$(SERVICE):$(VERSION) $(LOCAL_REGISTRY)/$(SERVICE):$(VERSION)
	docker push $(LOCAL_REGISTRY)/$(SERVICE):$(VERSION)

push_cloud: docker
	docker push $(MAINTAINER)/$(SERVICE):$(VERSION)