<template>
  <div>
    <!-- <label :class="{'--active': visible}">{{name}}</label> -->
    <re-filter-dropdown
      :class="{'highlighted' : visible}"
      class="dropdown--filter"
      :visible="visible"
      @visibility="onVisibility"
    >
      <span slot="dropdown-header">
        <span>{{name}}</span>
      </span>
      <div slot="dropdown-content">
        <input
          class="filter-options"
          type="text"
          v-model="searchText"
          autofocus
          placeholder="Search..."
        >
        <ul v-if="!multilevel">
          <li v-for="option in filterOptions(filter.values, searchText)" :key="option.id">
            <re-checkbox
              class="re-checkbox--dark"
              :value="option.selected"
              @change="onFilterChanged(filter.id, $event, option.id)"
            >{{option.name}} ({{option.count}})</re-checkbox>
          </li>
          <li v-if="!filterOptions(filter.values, searchText).length">0 results</li>
        </ul>
        <ul v-else>
          <li v-for="option in filterOptions(Object.keys(filter), searchText)" :key="option.id">
            <span class="filter-options__button" @click="showSecondLevel(option)" :class="[secondLevel !== undefined ? 'hidden' : '',secondLevel === option ? 'active' : '']">
              <span v-html="highlightSearch(option, searchText)"></span>
              <span class="filter-options__chev"></span>
            </span>
            <div class="second-level" :class="[secondLevel === option ? 'active' : '']">
              <span class="filter-options__back"><span class="filter-options__back__chev" @click="showFirstLevel()"></span>{{option}}</span>
              <input
                class="filter-options"
                type="text"
                v-model="searchTextValue"
                autofocus
                placeholder="Search..."
              >
              <ul>
                <li :class="[filtersStatus[option] ? '--selected' : '']" v-for="secondOption in filterOptions(filter[option].values, searchTextValue)" :key="secondOption.index">
                  <re-checkbox
                    class="re-checkbox--dark"
                    :value="secondOption.selected"
                     @change="onFilterChanged(option, $event, secondOption.id)"
                  ><span v-html="highlightSearch(secondOption.name, searchTextValue)"></span> {{secondOption.count}}</re-checkbox>
                </li>
                <li v-if="!filterOptions(filter[option].values, searchTextValue).length">0 results</li>
              </ul>
              <div class="filter__buttons">
                <re-button
                  class="button-tertiary--small button-tertiary--outline"
                  @click="onCancel"
                >Cancel</re-button>
                <re-button
                  class="button-primary--small"
                  @click="onApply"
                >Apply</re-button>
              </div>
            </div>
          </li>
          <li v-if="!filterOptions(Object.keys(filter), searchText).length">0 results</li>
        </ul>
        <div class="filter__buttons" v-if="!multilevel">
          <re-button
            class="button-tertiary--small button-tertiary--outline"
            @click="onCancel"
          >Cancel</re-button>
          <re-button
            class="button-primary--small"
            @click="onApply"
          >Apply</re-button>
        </div>
      </div>
    </re-filter-dropdown>
    <div class="overlay" v-if="visible"></div>
  </div>
</template>

<script>
import reButton from '@/components/elements/core/reButton/reButton';
import reFilterDropdown from '@/components/elements/core/reDropdown/reFilterDropdown';
import reCheckbox from '@/components/elements/core/reCheckbox/reCheckbox';

export default {
  name: 'select-filter',
  props: {
    filter: {
      type: Object,
    },
    name: {
      type: String,
    },
    multilevel: {
      type: Boolean,
      default: false,
    },
    filtersStatus: Object,
  },
  data: () => ({
    visible: false,
    searchText: undefined,
    searchTextValue: undefined,
    secondLevel: undefined,
  }),
  methods: {
    onFilterChanged(filterId, value, id) {
      this.$emit('filter-changed', filterId, id, value);
    },
    onVisibility(value) {
      this.visible = value;
      this.secondLevel = undefined;
      this.searchText = undefined;
      this.searchTextValue = undefined;
    },
    onApply() {
      this.$emit('apply');
      this.visible = false;
      this.secondLevel = undefined;
    },
    onCancel() {
      this.visible = false;
    },
    filterOptions(options, text) {
      if (text === undefined) {
        return options;
      }
      return options.filter((option) => {
        let filteroption;
        if (option.length) {
          filteroption = option.toString().toLowerCase().match(text.toLowerCase());
        } else if (option.name) {
          filteroption = option.name.toString().toLowerCase().match(text.toLowerCase());
        }
        return filteroption;
      });
    },
    highlightSearch(option, searchText) {
      if (!searchText) {
        return option;
      }
      return option.toString().replace(new RegExp(searchText, 'i'), match => `<span class="highlight-text">${match}</span>`);
    },
    showSecondLevel(option) {
      this.secondLevel = option;
    },
    showFirstLevel() {
      this.secondLevel = undefined;
      this.searchText = undefined;
      this.searchTextValue = undefined;
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
    reButton,
    reFilterDropdown,
    reCheckbox,
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
    .dropdown--open & {
      visibility: hidden
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
}
.filter-options {
  &__back{
    color: palette(purple);
    margin-top: 1em;
    display: flex;
    align-items: center;
    &__chev {
      cursor: pointer;
      margin-right: 1em;
      padding: 0.5em;
      &:after {
        content: '';
        border-color: palette(purple);
        border-style: solid;
        border-width: 1px 1px 0 0;
        display: inline-block;
        height: 8px;
        width: 8px;
        transform: rotate(-135deg);
        transition: all 1.5s ease;
        margin-bottom: 2px;
        margin-left: auto;
        margin-right: 0;
      }
    }
  }
  &__button {
    display: flex;
    cursor: pointer;
    min-width: 135px;
    transition: min-width 0.2s ease;
    &.active {
      min-width: 270px;
      transition: min-width 0.2s ease;
    }
    &.hidden {
      opacity: 0;
    }
  }
  &__chev {
    padding-left: 2em;
    margin-right: 0;
    margin-left: auto;
    background: none;
    border: none;
    outline: none;
    &:after {
      content: '';
      border-color: #4a4a4a;
      border-style: solid;
      border-width: 1px 1px 0 0;
      display: inline-block;
      height: 8px;
      width: 8px;
      transform: rotate(43deg);
      transition: all 1.5s ease;
      margin-bottom: 2px;
      margin-left: auto;
      margin-right: 0;
    }
  }
}
.second-level {
  position: absolute;
  top: -2px;
  padding: 0 1em;
  left: -2px;
  right: -2px;
  bottom: auto;
  background: palette(white);
  z-index: 1;
  min-width: 300px;
  max-height: none !important;
  min-height: 102%;
  border: 2px solid palette(purple);
  opacity: 0;
  pointer-events: none;
   ul {
    max-height: 240px;
    margin: 0 -1em 4em -1em;
  }
  .filter__buttons {
    position: absolute;
    bottom: 0;
    right: 1em;
  }
  & > * {
    opacity: 0;
    transition: all 0.3s ease;
  }
  &.active {
    opacity: 1;
    pointer-events: all;
    transition: all 0.1s ease 0.3s;
    & > * {
      opacity: 1;
      transition: all 1s ease 0.7s;
    }
  }
  ::v-deep .checkbox-label {
    white-space: normal;
    word-break: break-all;
  }
}
</style>
<style lang="scss">
.highlight-text {
    display: inline-block;
    // font-weight: 600;
    background: palette(purple, verylight);
}

</style>
