
brew install autoconf automake libtool protobuf

pushd .
git clone --depth=1 https://github.com/google/sentencepiece.git /tmp/
cd /tmp/sentencepiece
perl -i -pe 's/libtoolize/glibtoolize/' autogen.sh
./autogen.sh
./configure
make
make check
sudo make install
popd

rm -rf /tmp/sentencepiece