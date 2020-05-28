<template>
  <div class="empty" :class="{'centered': centered}">
    <span class="empty__content">
      <div class="empty__illustration" v-if="emptyIcon">
        <svgicon width="60" height="60" :name="emptyIcon"></svgicon>
      </div>
      <h1 class="empty__title">{{emptyTitle}}</h1>
      <h2 class="empty__subtitle">{{emptySubtitle}}</h2>
      <ul>
        <li v-for="item in actionsButtons" :key="item.id">
          <router-link
            class="button-primary"
            :to="item.route"
            v-on:click.native="$emit('setModal',true)"
          >
            <svgicon :name="emptyButtonIcon" width="20" height="auto"></svgicon>
            {{emptyButton}}
          </router-link>
        </li>
      </ul>
    </span>
  </div>
</template>

<script>
export default {
  name: 're-empty-list',
  props: ['emptyIcon', 'emptyTitle', 'emptySubtitle', 'emptyButton', 'emptyButtonIcon', 'actionsButtons', 'centered'],
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/empty.scss";
@import "@/assets/scss/components/buttons.scss";
</style>
