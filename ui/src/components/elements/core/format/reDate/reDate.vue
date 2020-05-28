<template>
  <span>{{date | moment('from', 'now')}}</span>
</template>

<script>
export default {
  name: 're-date',
  data: () => ({
  }),
  props: ['date'],
};
</script>
