<?PHP
$pdo=new PDO('mysql:host=localhost;dbname=final;charset=utf8','root','');    //連結資料庫，用PDO
$sql=$pdo->prepare('insert into commodity values(null,?,?,?,?,?)');
if($sql->execute([$_REQUEST['c_name'],$_REQUEST['price'],$_REQUEST['quantity'],$_REQUEST['describe'],$_REQUEST['url1']])){
    echo'新增成功';
}
else{
    echo'新增失敗';
}
?>