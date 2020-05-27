<template>
  <div>
    <!-- <label :class="{'--active': visible}">Sort by</label> -->
    <re-filter-dropdown
      :class="{'highlighted' : visible}"
      class="dropdown--filter dropdown--sortable"
      :visible="visible"
      @visibility="onVisibility"
    >
      <span slot="dropdown-header">
        <span
          v-if="optionSelected"
          class="dropdown__selectables"
        >{{selectedSortedBy.text}} {{sortedRange(selectedSortedBy, defaultSortedByDir)}}</span>
        <span v-else>Sort by</span>
      </span>
      <div slot="dropdown-content">
        <ul class="dropdown__list">
          <span v-for="option in sortedOption" :key="option.text">
            <div
              class="dropdown__list__item"
              @click="sort(option.filter, sortOption)"
              v-for="sortOption in sortType"
              :key="sortOption.index"
            >
              <li
                v-if="notSelectedOption(option, sortOption)"
              >{{option.text}} {{sortedRange(option, sortOption)}}</li>
            </div>
          </span>
        </ul>
      </div>
    </re-filter-dropdown>
    <div class="overlay" v-if="visible"></div>
  </div>
</template>

<script>
import reFilterDropdown from '@/components/elements/core/reDropdown/reFilterDropdown';

export default {
  name: 're-sort-list',
  data: () => ({
    visible: false,
    defaultSortedBy: undefined,
    defaultSortedByDir: 'asc',
    optionSelected: false,
  }),

  methods: {
    onVisibility(value) {
      this.visible = value;
    },
    sort(currentSort, currentSortDir) {
      this.visible = false;
      this.optionSelected = true
      this.defaultSortedBy = currentSort;
      this.defaultSortedByDir = currentSortDir;
      this.$emit('sort', currentSort, currentSortDir);
    },
    notSelectedOption(option, sortOption) {
      if (
        sortOption === this.defaultSortedByDir
        && option.filter === this.defaultSortedBy
      ) {
        return false;
      }
      return true;
    },
    sortedRange(by, byDir) {
      return (
        `${by.range[byDir === this.sortType[0] ? 0 : 1]
        } - ${
          by.range[byDir === this.sortType[0] ? 1 : 0]}`
      );
    },
  },
  computed: {
    sortedOption() {
      // TODO This should be passed as component props if we consider it as a component
      return [
        // { filter: 'gold', text: 'Text', range: ['A', 'Z'] },
        { filter: 'predicted', text: 'Predicted as', range: ['A', 'Z'] },
        { filter: 'confidence', text: 'Confidence', range: ['0', '100'] },
      ];
    },
    sortType() {
      return ['asc', 'desc'];
    },
    selectedSortedBy() {
      const key = Object.keys(this.sortedOption).find(
        k => this.sortedOption[k].filter === this.defaultSortedBy,
      );
      return this.sortedOption[key] || this.sortedOption[0];
    },
  },
  updated() {
    if (this.visible) {
      document.body.classList.add('--fixed');
    } else {
      document.body.classList.remove('--fixed');
    }
  },
  components: {
    // reButton,
    reFilterDropdown,
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/dropdown.scss";
@import "@/assets/scss/apps/classifier/filters.scss";
.dropdown {
  &__placeholder {
    display: none;
    .dropdown--open & {
      display: block;
    }
  }
  &__selectables {
    vertical-align: middle;
    display: inline-block;
    .dropdown--open & {
      display: none;
    }
    & + .dropdown__selectables {
      &:before {
        content: ",  ";
        margin-right: 2px;
      }
      &:after {
        content: "...";
        margin-left: -2px;
      }
    }
  }
  &__list {
    padding-right: 0;
    &__item {
      cursor: pointer;
      &:hover {
        color: palette(purple);
      }
    }
  }
}
</style>
