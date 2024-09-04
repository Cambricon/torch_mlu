#ifndef THMP_SERIALIZATION_INC
#define THMP_SERIALIZATION_INC

template <class io>
void THMPStorage_writeFileRaw(
    c10::StorageImpl* self,
    io fd,
    bool save_size,
    uint64_t element_size);

template <class io>
c10::intrusive_ptr<c10::StorageImpl> THMPStorage_readFileRaw(
    io fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);
#endif
