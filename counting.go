package boom

import (
	"bytes"
	"encoding/binary"
	"hash"
	"hash/fnv"
	"io"
	"math"
)

// CountingBloomFilter implements a Counting Bloom Filter as described by Fan,
// Cao, Almeida, and Broder in Summary Cache: A Scalable Wide-Area Web Cache
// Sharing Protocol:
//
// http://pages.cs.wisc.edu/~jussara/papers/00ton.pdf
//
// A Counting Bloom Filter (CBF) provides a way to remove elements by using an
// array of n-bit buckets. When an element is added, the respective buckets are
// incremented. To remove an element, the respective buckets are decremented. A
// query checks that each of the respective buckets are non-zero. Because CBFs
// allow elements to be removed, they introduce a non-zero probability of false
// negatives in addition to the possibility of false positives.
//
// Counting Bloom Filters are useful for cases where elements are both added
// and removed from the data set. Since they use n-bit buckets, CBFs use
// roughly n-times more memory than traditional Bloom filters.
type CountingBloomFilter struct {
	buckets     *Buckets    // filter data
	hash        hash.Hash64 // hash function (kernel for all k functions)
	m           uint        // number of buckets
	k           uint        // number of hash functions
	count       uint        // number of items in the filter
	indexBuffer []uint      // buffer used to cache indices
}

// NewCountingBloomFilter creates a new Counting Bloom Filter optimized to
// store n items with a specified target false-positive rate and bucket size.
// If you don't know how many bits to use for buckets, use
// NewDefaultCountingBloomFilter for a sensible default.
func NewCountingBloomFilter(n uint, b uint8, fpRate float64) *CountingBloomFilter {
	var (
		m = OptimalM(n, fpRate)
		k = OptimalK(fpRate)
	)
	return &CountingBloomFilter{
		buckets:     NewBuckets(m, b),
		hash:        fnv.New64(),
		m:           m,
		k:           k,
		indexBuffer: make([]uint, k),
	}
}

// NewDefaultCountingBloomFilter creates a new Counting Bloom Filter optimized
// to store n items with a specified target false-positive rate. Buckets are
// allocated four bits.
func NewDefaultCountingBloomFilter(n uint, fpRate float64) *CountingBloomFilter {
	return NewCountingBloomFilter(n, 4, fpRate)
}

// Capacity returns the Bloom filter capacity, m.
func (c *CountingBloomFilter) Capacity() uint {
	return c.m
}

// K returns the number of hash functions.
func (c *CountingBloomFilter) K() uint {
	return c.k
}

// Count returns the number of items in the filter.
func (c *CountingBloomFilter) Count() uint {
	return c.count
}

// Test will test for membership of the data and returns true if it is a
// member, false if not. This is a probabilistic test, meaning there is a
// non-zero probability of false positives and false negatives.
func (c *CountingBloomFilter) Test(data []byte) bool {
	lower, upper := hashKernel(data, c.hash)

	// If any of the K bits are not set, then it's not a member.
	for i := uint(0); i < c.k; i++ {
		if c.buckets.Get((uint(lower)+uint(upper)*i)%c.m) == 0 {
			return false
		}
	}

	return true
}

// Same as Test but returns an approximate count of occurrences of data in CountingBloomFilter.
// Note that the count is always the minimum count, which assumes that no element removals are made.
// If element removal is done leading to false negatives, a different count metric should be used.
func (c *CountingBloomFilter) TestCount(data []byte) uint32 {
	lower, upper := hashKernel(data, c.hash)

	var ans uint32 = math.MaxUint32
	// Take the smallest count
	for i := uint(0); i < c.k; i++ {
		tmp := c.buckets.Get((uint(lower) + uint(upper)*i) % c.m)

		if tmp < ans {
			ans = tmp
		}
	}

	return ans
}

// Add will add the data to the Bloom filter. It returns the filter to allow
// for chaining.
func (c *CountingBloomFilter) Add(data []byte) Filter {
	lower, upper := hashKernel(data, c.hash)

	// Set the K bits.
	for i := uint(0); i < c.k; i++ {
		c.buckets.Increment((uint(lower)+uint(upper)*i)%c.m, 1)
	}

	c.count++
	return c
}

// TestAndAdd is equivalent to calling Test followed by Add. It returns true if
// the data is a member, false if not.
func (c *CountingBloomFilter) TestAndAdd(data []byte) bool {
	lower, upper := hashKernel(data, c.hash)
	member := true

	// If any of the K bits are not set, then it's not a member.
	for i := uint(0); i < c.k; i++ {
		idx := (uint(lower) + uint(upper)*i) % c.m
		if c.buckets.Get(idx) == 0 {
			member = false
		}
		c.buckets.Increment(idx, 1)
	}

	c.count++
	return member
}

// TestAndRemove will test for membership of the data and remove it from the
// filter if it exists. Returns true if the data was a member, false if not.
func (c *CountingBloomFilter) TestAndRemove(data []byte) bool {
	lower, upper := hashKernel(data, c.hash)
	member := true

	// Set the K bits.
	for i := uint(0); i < c.k; i++ {
		c.indexBuffer[i] = (uint(lower) + uint(upper)*i) % c.m
		if c.buckets.Get(c.indexBuffer[i]) == 0 {
			member = false
		}
	}

	if member {
		for _, idx := range c.indexBuffer {
			c.buckets.Increment(idx, -1)
		}
		c.count--
	}

	return member
}

// Reset restores the Bloom filter to its original state. It returns the filter
// to allow for chaining.
func (c *CountingBloomFilter) Reset() *CountingBloomFilter {
	c.buckets.Reset()
	c.count = 0
	return c
}

// SetHash sets the hashing function used in the filter.
// For the effect on false positive rates see: https://github.com/tylertreat/BoomFilters/pull/1
func (c *CountingBloomFilter) SetHash(h hash.Hash64) {
	c.hash = h
}

// WriteTo writes a binary representation of the CountingBloomFilter to an i/o stream.
// It returns the number of bytes written.
func (c *CountingBloomFilter) WriteTo(stream io.Writer) (int64, error) {
	err := binary.Write(stream, binary.BigEndian, uint64(c.count))
	if err != nil {
		return 0, err
	}
	err = binary.Write(stream, binary.BigEndian, uint64(c.m))
	if err != nil {
		return 0, err
	}
	err = binary.Write(stream, binary.BigEndian, uint64(c.k))
	if err != nil {
		return 0, err
	}

	// Create + insert index buffer:
	indexBuffer := make([]uint64, c.k)
	for i := uint(0); i < c.k; i++ {
		indexBuffer[i] = uint64(c.indexBuffer[i])
	}

	err = binary.Write(stream, binary.BigEndian, indexBuffer)
	if err != nil {
		return 0, err
	}

	writtenSize, err := c.buckets.WriteTo(stream)
	if err != nil {
		return 0, err
	}

	return writtenSize + int64((3+int(c.k))*binary.Size(uint64(0))), err
}

// ReadFrom reads a binary representation of BloomFilter (such as might
// have been written by WriteTo()) from an i/o stream. It returns the number
// of bytes read.
func (c *CountingBloomFilter) ReadFrom(stream io.Reader) (int64, error) {
	var count, m, k uint64
	var buckets Buckets

	err := binary.Read(stream, binary.BigEndian, &count)
	if err != nil {
		return 0, err
	}
	err = binary.Read(stream, binary.BigEndian, &m)
	if err != nil {
		return 0, err
	}
	err = binary.Read(stream, binary.BigEndian, &k)
	if err != nil {
		return 0, err
	}

	indexBuffer := make([]uint64, k)

	// TODO: Explicitly verify buffer lengths are the same?
	err = binary.Read(stream, binary.BigEndian, &indexBuffer)
	if err != nil {
		return 0, err
	}

	readSize, err := buckets.ReadFrom(stream)
	if err != nil {
		return 0, err
	}

	c.count = uint(count)
	c.m = uint(m)
	c.k = uint(k)
	c.indexBuffer = make([]uint, k)

	// Convert index buffer elements to uint:
	for i := uint(0); i < c.k; i++ {
		c.indexBuffer[i] = uint(indexBuffer[i])
	}

	c.buckets = &buckets
	return readSize + int64((3+int(c.k))*binary.Size(uint64(0))), nil
}

// GobEncode implements gob.GobEncoder interface.
func (c *CountingBloomFilter) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	_, err := c.WriteTo(&buf)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// GobDecode implements gob.GobDecoder interface.
func (c *CountingBloomFilter) GobDecode(data []byte) error {
	buf := bytes.NewBuffer(data)
	_, err := c.ReadFrom(buf)
	// TODO: encode hash also maybe?
	if c.hash == nil {
		c.hash = fnv.New64()
	}

	return err
}
