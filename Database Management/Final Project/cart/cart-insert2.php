<?PHP
$pdo=new PDO('mysql:host=localhost;dbname=final;charset=utf8','root','');    //連結資料庫，用PDO
$sql=$pdo->prepare('insert into cart values(?,?,?,?,?)');
if($sql->execute([$_REQUEST['thing_id'],$_REQUEST['name'],$_REQUEST['price'],$_REQUEST['amount'],$_REQUEST['url1']])){
?>
<script>
          window.location = '../view/final2.php'
</script>
<?php
}
?>