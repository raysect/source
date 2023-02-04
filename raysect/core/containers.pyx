# cython: language_level=3

# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


cdef class _Item:
    """
    Internal item class for holding individual LinkedList items with references to neighbors.

    :param _Item previous: Reference to the previous container item, will be None if this is the first item.
    :param object value: The object to be stored as this item value.
    :param _Item next_item: Reference to the next container item, will be None if this is the last item.
    """

    def __init__(self, _Item previous, object value, _Item next_item=None):

        self.previous = previous
        self.value = value
        self.next = next_item


cdef class LinkedList:
    """
    Basic implementation of a Linked List for fast container operations in cython.

    :param object initial_items: Optional iterable for initialising container.

    :ivar int length: number of items in the container
    :ivar first: starting element of container
    :ivar last: final element of container
    """

    def __init__(self, initial_items=None):

        self.length = 0
        self.first = None
        self.last = None

        if initial_items:
            self.add_items(initial_items)

    def __getitem__(self, item):

        if not isinstance(item, int):
            raise ValueError('Container index must be an int.')
        return self.get_index(item)

    def __iter__(self):

        cdef:
            int i = 0
            _Item current_item

        if not self.length:
            return

        current_item = self.first
        while i != self.length:
            yield current_item.value
            current_item = current_item.next
            i += 1

    cpdef bint is_empty(self):
        """ Returns True if the container is empty. """

        if self.length:
            return False
        else:
            return True

    cpdef add(self, object value):
        """ Add an item to the end of the container.

        :param object value: The item to add to the end of the container.
        """

        cdef _Item new_item

        new_item = _Item(self.last, value)

        if self.first:
            self.last.next = new_item
            self.last = new_item
        else:
            self.first = new_item
            self.last = new_item
        self.length += 1

    cpdef add_items(self, object iterable):
        """ Extend this container with another iterable container.

        :param object iterable: Iterable object such as a list or ndarray with
          which to extend this container.
        """

        for item in iterable:
            self.add(item)

    cpdef object get_index(self, int index):
        """
        Get the item from the container at specified index.

        :param int index: requested item index
        """

        cdef:
            int i = 0
            _Item current_item

        if not self.length:
            raise RuntimeError("Get_index operation cannot be performed because container is empty.")

        if index >= self.length:
            raise IndexError("Specified Index is longer than the container. Get_index operation cannot be performed.")

        current_item = self.first
        while i != index:
            current_item = current_item.next
            i += 1

        return current_item.value

    cpdef insert(self, object value, int index):
        """
        Insert an item at the specified index.

        :param object value: item to insert
        :param int index: index at which to insert this item
        """

        cdef:
            int i = 0
            _Item new_item, previous_item, current_item

        if not self.length:
            raise RuntimeError("Insert operation is not supported on empty containers.")

        if index >= self.length:
            raise IndexError("Specified Index is longer than the container. Insert operation cannot be performed.")

        previous_item = None
        current_item = self.first

        while i != index:
            previous_item = current_item
            current_item = current_item.next
            i += 1

        new_item = _Item(previous_item, value, next_item=current_item)

        if previous_item:
            previous_item.next = new_item
        if current_item:
            current_item.previous = new_item
        if index == 0:
            self.first = new_item
        elif index == self.length - 1:
            self.last = new_item

        self.length += 1

    cpdef object remove(self, int index):
        """
        Remove and return the specified item from the container.

        :param int index: Index at which an item will be removed.
        :return: The object at the specified index position.
        """

        cdef:
            int i = 0
            _Item new_item, previous_item, current_item, next_item

        if not self.length:
            raise RuntimeError("Remove operation is not supported on empty containers.")

        if index >= self.length:
            raise IndexError("Specified Index is longer than the container. Remove operation cannot be performed.")

        previous_item = None
        current_item = self.first
        while i != index:
            previous_item = current_item
            current_item = current_item.next
            i += 1

        next_item = current_item.next

        if next_item:
            previous_item.next = next_item
            next_item.previous = previous_item
        else:
            previous_item.next = None

        self.length -= 1

        return current_item.value


cdef class Stack(LinkedList):
    """
    Basic implementation of a Stack container for fast container operations in cython.
    Inherits attributes and methods from LinkedList.
    """

    cpdef push(self, object value):
        """ Adds an item to the top of the stack

        :param object value: Object that will be pushed to top of the stack
        """

        cdef _Item last, new_item

        last = self.last

        if last:
            new_item = _Item(last, value)
            last.next = new_item
        else:
            new_item = _Item(None, value)

        self.last = new_item

        self.length += 1

    cpdef object pop(self):
        """ Removes and returns the most recently added item from the stack

        :rtype: object
        """

        cdef _Item last, previous

        # If container is empty, return None value
        if not self.length:
            return None

        # Get the last value from the stack, reset the container's last position with new end point.
        last = self.last
        previous = last.previous
        previous.next = None
        self.last = previous

        self.length -= 1

        return last.value


cdef class Queue(LinkedList):

    """
    Basic implementation of a Queue container for fast container operations in cython.
    Inherits attributes and methods from LinkedList.
    """

    cpdef object next_in_queue(self):
        """ Returns the next object in the queue

        :rtype: object
        """

        cdef _Item first, second

        # If container is empty, return None value
        if not self.length:
            return None

        first = self.first
        second = first.next

        if second:
            self.first = second
            second.previous = None
        else:
            self.first = None
            self.last = None

        self.length -= 1

        return first.value


