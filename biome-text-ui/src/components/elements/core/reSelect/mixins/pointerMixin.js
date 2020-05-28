export default {
  data() {
    return {
      pointer: 0,
      pointerDirty: false,
    };
  },
  props: {
    /**
     * Enable/disable highlighting of the pointed value.
     * @type {Boolean}
     * @default true
     */
    showPointer: {
      type: Boolean,
      default: true,
    },
    optionHeight: {
      type: Number,
      default: 40,
    },
  },
  computed: {
    pointerPosition() {
      return this.pointer * this.optionHeight;
    },
    visibleElements() {
      return this.optimizedHeight / this.optionHeight;
    },
  },
  watch: {
    filteredOptions() {
      this.pointerAdjust();
    },
    isOpen() {
      this.pointerDirty = false;
    },
  },
  methods: {
    optionHighlight(index, option) {
      return {
        'select__option--highlight': index === this.pointer && this.showPointer,
        'select__option--selected': this.isSelected(option),
      };
    },
    groupHighlight(index, selectedGroup) {
      if (!this.groupSelect) {
        return ['select__option--disabled'];
      }

      const group = this.options.find(option => option[this.groupLabel] === selectedGroup.$groupLabel);

      return [
        this.groupSelect ? 'select__option--group' : 'select__option--disabled',
        { 'select__option--highlight': index === this.pointer && this.showPointer },
        { 'select__option--group-selected': this.wholeGroupSelected(group) },
      ];
    },
    addPointerElement({ key } = 'Enter') {
      /* istanbul ignore else */
      if (this.filteredOptions.length > 0) {
        this.select(this.filteredOptions[this.pointer], key);
      }
      this.pointerReset();
    },
    pointerForward() {
      /* istanbul ignore else */
      if (this.pointer < this.filteredOptions.length - 1) {
        this.pointer += 1;
        /* istanbul ignore next */
        if (this.$refs.list.scrollTop <= this.pointerPosition - (this.visibleElements - 1) * this.optionHeight) {
          this.$refs.list.scrollTop = this.pointerPosition - (this.visibleElements - 1) * this.optionHeight;
        }
        /* istanbul ignore else */
        if (
          this.filteredOptions[this.pointer]
          && this.filteredOptions[this.pointer].$isLabel
          && !this.groupSelect
        ) this.pointerForward();
      }
      this.pointerDirty = true;
    },
    pointerBackward() {
      if (this.pointer > 0) {
        this.pointer -= 1;
        /* istanbul ignore else */
        if (this.$refs.list.scrollTop >= this.pointerPosition) {
          this.$refs.list.scrollTop = this.pointerPosition;
        }
        /* istanbul ignore else */
        if (
          this.filteredOptions[this.pointer]
          && this.filteredOptions[this.pointer].$isLabel
          && !this.groupSelect
        ) this.pointerBackward();
      } else if (this.filteredOptions[this.pointer] && this.filteredOptions[0].$isLabel && !this.groupSelect) {
        this.pointerForward();
      }
      this.pointerDirty = true;
    },
    pointerReset() {
      /* istanbul ignore else */
      if (!this.closeOnSelect) return;
      this.pointer = 0;
      /* istanbul ignore else */
      if (this.$refs.list) {
        this.$refs.list.scrollTop = 0;
      }
    },
    pointerAdjust() {
      /* istanbul ignore else */
      if (this.pointer >= this.filteredOptions.length - 1) {
        this.pointer = this.filteredOptions.length
          ? this.filteredOptions.length - 1
          : 0;
      }

      if (this.filteredOptions.length > 0
        && this.filteredOptions[this.pointer].$isLabel
        && !this.groupSelect
      ) {
        this.pointerForward();
      }
    },
    pointerSet(index) {
      this.pointer = index;
      this.pointerDirty = true;
    },
  },
};
